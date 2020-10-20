"""Generate cache for optimal configurations
"""
import logging, os, json
import numpy as np


logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Cache():
    """Cache generator"""
    def __init__(self, filename, prefix=None):
        """Initialize cache generator

        Args:
            filename : Path of the file
            prefix   : Prefix to include on filenames when saving

        """
        self.filename = filename
        self.prefix   = "" if prefix is None else prefix + "_"
        self._set_paths()

    def _set_paths(self):
        """Define paths to save the configuration files

        Set the cache filename based on the name of the dataset and the
        identifier passed as argument.

        """
        this_file       = os.path.realpath(__file__)
        rootdir         = os.path.dirname(os.path.dirname(this_file))
        no_ext_ds_name  = os.path.splitext(os.path.basename(self.filename))[0]
        ds_name         = no_ext_ds_name.replace("-comp", "")
        cache_path      = os.path.join(rootdir, 'cache')
        self.cache_path = os.path.join(cache_path, ds_name)

        # Create the folder if it doesn't exist
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)

    def _is_cache_available(self, cache_file):
        """Check if the cache file is available"""
        return os.path.isfile(cache_file)

    def save(self, data, identifier):
        """Save cache into a JSON file

        Args:
            data       : Data structure to save into JSON file
            identifier : Unique name used to identify the information that is
                         being cached. It will used as the suffix of the
                         resulting cache filename.

        """
        cache_file = os.path.join(self.cache_path,
                                  self.prefix + "{}.json".format(identifier))

        with open(cache_file, 'w') as fd:
            json.dump(data, fd, cls=NumpyEncoder)

        logger.info("Saved {} cache on {}".format(identifier, cache_file))

    def load(self, identifier):
        """Load cache from JSON file

        Args:
            identifier : Unique name used to identify the information that is
                         being cached. It will used as the suffix of the
                         resulting cache filename.x

        """
        cache_file = os.path.join(self.cache_path,
                                  self.prefix + "{}.json".format(identifier))
        cached_data = None

        if (self._is_cache_available(cache_file)):
            with open(cache_file) as fd:
                cached_data = json.load(fd)

            logger.info("Loaded {} cache from {}".format(identifier,
                                                         cache_file))
        return cached_data


