"""Download dataset from server
"""
import subprocess, os, logging, json
from ptp import util

logger = logging.getLogger(__name__)


class Download():
    def __init__(self, dataset):
        """Download dataset from dataset server

        Args:
            dataset : dataset name/path

        Returns:
            Boolean indicating whether file was found and downloaded

        """
        this_file          = os.path.realpath(__file__)
        rootdir            = os.path.dirname(os.path.dirname(this_file))
        local_repo         = os.path.join(rootdir, "data")
        self.ds_name       = os.path.basename(dataset)
        no_ext_ds_name     = os.path.splitext(self.ds_name)[0]
        self.local_ds_path = os.path.join(local_repo, self.ds_name)
        comp_ds_name       = no_ext_ds_name + "-comp"
        comp_exts          = [".xz", ".pbz2", ".gz", ".pickle", ".json"]
        self.comp_ds_paths = [os.path.join(local_repo, comp_ds_name + e) for
                              e in comp_exts]

        home          = os.path.expanduser("~")
        cfg_path      = os.path.join(home, ".ptp")
        self.cfg_file = os.path.join(cfg_path, "config.json")
        self._check_cfg_path(cfg_path)

        self.cfg = self._load_cfg()
        if (self.cfg is None):
            logger.error("Failed to load dataset server configurations")
            return False

    def _check_cfg_path(self, path):
        """Check if path to cfg folder exists or create it otherwise"""
        if (not os.path.exists(path)):
            os.mkdir(path)
        elif (not os.path.isdir(path)):
            raise IsADirectoryError(
                "{} already exists, but is not a directory".format(path))

    def _load_cfg(self):
        """Load user credentials from configuration file"""
        if (os.path.exists(self.cfg_file)):
            with open(self.cfg_file) as fd:
                cfg = json.load(fd)
            logger.info("Loaded dataset server configurations from {}".format(
                self.cfg_file))
        else:
            logger.info("Couldn't find access information for dataset server.")
            cfg = self._create_cfg()

        return cfg

    def _create_cfg(self):
        """Create configuration file with user-provided credentials"""
        if (not util.ask_yes_or_no("Provide information now?")):
            return

        cfg  = list()
        more = True
        while (more):
            server = input("IP address of the dataset server: ")
            path   = input("Path to dataset repository on server: ")
            user   = input("Username to access the server: ")
            cfg.append({
                'addr' : server,
                'path' : path,
                'user' : user
            })
            more = util.ask_yes_or_no("Add another address?")

        with open(self.cfg_file, 'w') as fd:
            json.dump(cfg, fd)

        logger.info(f"Saved dataset server configurations on {self.cfg_file}")
        return cfg

    def _download(self, scp_addr):
        """Download dataset from dataset server

        Args:
            ds_name  : dataset file name
            scp_addr : url of dataset repository on remote server to scp from

        Returns:
            Path to the file that was downloaded. None if not found.

        """
        no_ext_ds_name = os.path.splitext(self.ds_name)[0]
        ds_prefix      = no_ext_ds_name.replace("-comp", "")
        ds_suffixes    = ["-comp.xz", "-comp.pbz2", "-comp.gz", "-comp.pickle",
                          "-comp.json", ".json"]
        ds_names       = [ds_prefix + suffix for suffix in ds_suffixes]

        # Try to load in order of compression (most compressed first)
        found = False
        for ds_name in ds_names:
            cmd = ["scp", os.path.join(scp_addr, ds_name), "data/"]
            try:
                logger.info("> %s" %(" ".join(cmd)))
                out   = subprocess.check_output(cmd)
                found = True
            except subprocess.CalledProcessError as e:
                pass

            if (found):
                print("Downloaded {} from {}".format(ds_name, scp_addr))
                return "data/" + ds_name
            else:
                logger.info("Couldn't find file {} in {}".format(
                    ds_name, scp_addr))

    def run(self):
        """Execute download

        Returns:
            Path to dataset file

        """

        # Does the dataset exist on the local repository already?
        if (os.path.exists(self.local_ds_path)):
            logger.info("Dataset already available locally")
            return self.local_ds_path

        # Search for a compressed dataset locally
        if ("-comp" not in self.local_ds_path):
            for path in self.comp_ds_paths:
                if (os.path.exists(path)):
                    logger.info("Found compressed dataset {}".format(path))
                    return path

        print("Dataset not available locally. Try to download from server.")

        for entry in self.cfg:
            addr = entry['user'] + "@" + entry['addr'] + ":" + entry['path']
            path = self._download(addr)
            if (path is not None):
                return path

        # We should have returned before if a datset was found on the local or
        # remote repositories
        raise RuntimeError("Couldn't find dataset")


