"""Compress/decompress PTP datasets"""
import json, pickle, gzip, bz2, lzma
import logging, time, os


class Codec():
    def __init__(self, ds={}, filename="", compressed=False):
        """Construct codec

        Args:
            ds         : (dictionary) dataset
            filename   : (string) dataset file name
            compressed : (bool) whether the supplied dataset is compressed
        """
        assert(isinstance(ds, dict))
        assert(isinstance(filename, str))

        # When the dataset is not provided to the constructor, it means we need
        # to load it from the file
        if (ds == {}):
            assert(filename != ""), \
                "Please provide dataset file name when the dataset itself is \
                not provided"
            ext = os.path.splitext(filename)[1]
            assert(ext in [".json", ".pickle", ".gz", ".pbz2", ".xz"]), \
                "File extension {} not supported".format(ext)
            self.compressed = "-comp" in filename
            self._load(filename)
        else:
            self.ds         = ds # mutable and a shallow copy
            self.compressed = compressed

        if (filename != ""):
            self.orig_size = os.path.getsize(filename)
            no_ext_name    = os.path.splitext(filename)[0]
            self.name      = no_ext_name
            self.out_name  = no_ext_name + "-comp"
        else:
            self.orig_size = self.name = self.out_name = None

    def _load(self, filename):
        """Load dataset from file"""
        logger = logging.getLogger("codec-load")
        tic = time.time()
        if (filename.endswith('.json')):
            with open(filename, 'r') as fd:
                self.ds = json.load(fd)
        elif (filename.endswith('.pickle')):
            with open(filename, 'rb') as fd:
                self.ds = pickle.load(fd)
        elif (filename.endswith('.gz')):
            with gzip.open(filename, "rb") as fd:
                self.ds = pickle.load(fd)
        elif (filename.endswith('.pbz2')):
            with bz2.BZ2File(filename, 'rb') as fd:
                self.ds = pickle.load(fd)
        elif (filename.endswith('.xz')):
            with lzma.open(filename, "rb") as fd:
                self.ds = pickle.load(fd)
        else:
            raise ValueError("Unsupported extension {}".format(ext))
        toc = time.time()
        logger.debug("Deserialization took {:5.2f} secs".format(toc - tic))

    def compress(self):
        """Reorganize dataset more efficiently for storing into files

        The data['data'] member of the dataset holds a list of dictionaries,
        each containing several metrics. This is is very inefficient for
        storage, since the keys (strings) are repeated on every dictionary.

        Some of the metrics in the dataset are present on all
        dictionaries. Hence, they can be stored in lists directly. Other metrics
        are not present in all dictionaries, in which case they should be stored
        in a pair of lists, one containing the actual time-series, the other
        containing the indexes where the elements are present in the dataset.

        Args:
            data : Dataset dictionary formatted as {'metadata' : x, 'data' : y},
                   i.e.\ as a dictionary containing the metadata and data keys.

        Returns:
            (dict) The compressed dataset

        """
        logger = logging.getLogger("codec-compress")
        logger.info("Compress dataset")

        if (self.compressed):
            logger.warning("Dataset is already compressed")
            return self.ds

        self.ds['indexed']     = {}
        self.ds['idx']         = {}
        self.ds['non-indexed'] = {}

        # Find all possible dictionary keys (no need to loop over the entire
        # dataset, they should be present on the first entries)
        keys  = set()
        for x in self.ds['data']:
            keys.update(x.keys())

        # Find the keys that are always present
        candidate_keys      = self.ds['data'][0].keys()
        always_present_keys = list()
        for key in candidate_keys:
            always_present = all([key in x for x in self.ds['data']])
            if always_present:
                always_present_keys.append(key)

        # For keys that are always present, move to non-indexed lists and remove
        # from the original dictionaries in self.ds['data']
        for key in sorted(always_present_keys):
            logger.debug("Non-indexed key {}".format(key))
            ts = [x[key] for x in self.ds['data']]
            self.ds['non-indexed'][key] = ts
            for x in self.ds['data']:
                x.pop(key, None)

        # For keys that are not always present, move both the time-series and
        # the indexes into the indexed lists
        sporadic_keys = keys.difference(set(always_present_keys))
        for key in sorted(sporadic_keys):
            logger.debug("Indexed key {}".format(key))
            ts  = [x[key] for x in self.ds['data'] if key in x]
            idx = [i for i,x in enumerate(self.ds['data']) if key in x]
            # Save time-series
            self.ds['indexed'][key] = ts
            # Save indexes
            #
            # First, check if another time-series has the same indexes. In the
            # positive case, save the index entry as a string (the key of the
            # other time-series with equal indexes). This avoids saving two
            # equal vectors of indexes unnecessarily.
            idx_vec_exists = False
            for idx_key in self.ds['idx']:
                if (idx == self.ds['idx'][idx_key]):
                    self.ds['idx'][key] = idx_key
                    idx_vec_exists = True
                    break

            if (not idx_vec_exists):
                self.ds['idx'][key] = idx

            for x in self.ds['data']:
                x.pop(key, None)

        # We should have removed all elements from all dictionaries
        assert(not any([x for x in self.ds['data']]))
        self.ds.pop('data')
        # Maybe some index vectors were repeated and further savings were
        # achieved
        logger.debug("Unique index vectors:")
        for key in self.ds['idx']:
            if (isinstance(self.ds['idx'][key], list)):
                logger.debug("{}".format(key))
        logger.debug("Repeated index vectors:")
        for key in self.ds['idx']:
            if (isinstance(self.ds['idx'][key], str)):
                logger.debug("{} -> {}".format(key, self.ds['idx'][key]))

        self.compressed = True
        return self.ds

    def decompress(self):
        """Revert the compression

        Returns:
            (dict) The decompressed dataset

        """
        logger = logging.getLogger("codec-decompress")
        logger.info("Decompress dataset")

        if (not self.compressed):
            logger.warning("Dataset is already decompressed")
            return self.ds

        assert('data' not in self.ds)

        # Non-indexed time-series
        for key in self.ds['non-indexed']:
            logger.debug("Non-indexed key {}".format(key))
            # Initialize the list of dictionaries
            if ('data' not in self.ds):
                ds_len = len(self.ds['non-indexed'][key])
                self.ds['data'] = [{} for _ in range(ds_len)]
            # Add values to each dictionary
            for i,x in enumerate(self.ds['non-indexed'][key]):
                self.ds['data'][i][key] = x
        self.ds.pop('non-indexed')

        # Indexed time-series
        assert('data' in self.ds)
        for key in self.ds['indexed']:
            logger.debug("Indexed key {}".format(key))
            ts  = self.ds['indexed'][key]
            if (isinstance(self.ds['idx'][key], str)):
                idx = self.ds['idx'][self.ds['idx'][key]]
            else:
                idx = self.ds['idx'][key]
            # Add values to each dictionary
            for i,x in zip(idx, ts):
                self.ds['data'][i][key] = x
        self.ds.pop('indexed')
        self.ds.pop('idx')
        assert(len(self.ds.keys()) == 2)
        assert("metadata" in self.ds and "data" in self.ds)

        self.compressed = False
        return self.ds

    def dump(self, ext="xz"):
        """Dump dataset to file

        Args:
            ext : Output file extension, which also determines the binary
                  compression scheme to be adopted. Choose from "json",
                  "pickle", "gz", "pbz2" or "xz"

        """
        logger = logging.getLogger("codec-dump")

        assert(ext in ["json", "pickle", "gz", "pbz2", "xz"])
        assert(self.compressed), "Dataset has not been compressed yet"
        assert(self.out_name is not None), "Original dataset name not provided"

        ext = ext.lower()
        outfile = "{}.{}".format(self.out_name, ext)

        logger.info("Dump compressed dataset into {}".format(outfile))

        tic = time.time()
        if ext == "json":
            with open(outfile, 'w') as fd:
                json.dump(self.ds, fd)
        elif ext == "pickle":
            with open(outfile, 'wb') as fd:
                pickle.dump(self.ds, fd)
        elif ext == "gz":
            with gzip.open(outfile, "wb") as fd:
                pickle.dump(self.ds, fd)
        elif ext == "pbz2":
            with bz2.BZ2File(outfile, 'wb') as fd:
                pickle.dump(self.ds, fd)
        elif ext == "xz":
            with lzma.open(outfile, "wb") as fd:
                pickle.dump(self.ds, fd)
        else:
            raise ValueError("Unsupported extension {}".format(ext))
        toc = time.time()

        duration    = (toc - tic)
        new_size    = os.path.getsize(outfile)
        new_size_mb = new_size / (2**20)

        if (self.orig_size is not None):
            ratio = self.orig_size / new_size
            logger.info("Compression: format: {:6s} - size: {:5.2f} MB - "
                        "ratio - {:5.2f} - duration: {:5.2f} secs".format(
                            ext, new_size_mb, ratio, duration))
        else:
            logger.info("Compression: format: {:6s} - size: {:5.2f} MB - "
                        "duration: {:5.2f} secs".format(
                            ext, new_size_mb, duration))

