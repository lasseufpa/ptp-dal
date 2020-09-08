"""Generate cache for optimal configurations
"""
import logging, os, json
logger = logging.getLogger(__name__)


class Cache():
    """Cache generator"""
    def __init__(self, filename):
        """Initialize cache generator

        Args:
            filename : Path of the file

        """
        self.filename = filename
        self._set_paths()

    def _set_paths(self):
        """Define paths to save the configuration files

        Set the cache filename based on the name of the dataset and the
        identifier passed as argument.

        """
        this_file       = os.path.realpath(__file__)
        rootdir         = os.path.dirname(os.path.dirname(this_file))
        no_ext_ds_name  = os.path.splitext(os.path.basename(self.filename))[0]
        self.ds_name    = no_ext_ds_name.replace("-comp", "")
        cache_path      = os.path.join(rootdir, 'cache')
        self.cache_path = os.path.join(cache_path, self.ds_name)
        self.basename   = os.path.join(self.cache_path, self.ds_name)

        # Create the folder if it doesn't exist
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)

    def _is_cache_available(self):
        """Check if the cache file is available"""
        return os.path.isfile(self.cache_file)

    def save(self, data, identifier):
        """Save cache into a JSON file

        Args:
            data       : Data structure to save into JSON file
            identifier : Unique name used to identify the information that is
                         being cached. It will used as the suffix of the
                         resulting cache filename.

        """
        self.cache_file = self.basename + "-{}.json".format(identifier)

        with open(self.cache_file, 'w') as fd:
            json.dump(data, fd)

        logging.info("Saved {} cache on {}".format(identifier, self.cache_file))

    def load(self, identifier):
        """Load cache from JSON file

        Args:
            identifier : Unique name used to identify the information that is
                         being cached. It will used as the suffix of the
                         resulting cache filename.x

        """
        self.cache_file = self.basename + f'-{identifier}.json'
        cached_data     = None

        if (self._is_cache_available()):
            with open(self.cache_file) as fd:
                cached_data = json.load(fd)

            logging.info("Loaded {} cache from {}".format(identifier,
                                                          self.cache_file))
        return cached_data


