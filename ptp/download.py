"""Download dataset from server
"""
import subprocess, os, logging, json
from ptp import util


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
        self.local_ds_path = os.path.join(local_repo, self.ds_name)

        home          = os.path.expanduser("~")
        cfg_path      = os.path.join(home, ".ptp")
        self.cfg_file = os.path.join(cfg_path, "config.json")
        self._check_cfg_path(cfg_path)

        self.cfg = self._load_cfg()
        if (self.cfg is None):
            logging.error("Failed to load dataset server configurations")
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
            logging.info("Loaded dataset server configurations from {}".format(
                self.cfg_file))
        else:
            logging.info("Couldn't find access information for dataset server.")
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

        logging.info(f"Saved dataset server configurations on {self.cfg_file}")
        return cfg

    def _download(self, ds_name, scp_addr):
        """Download dataset from dataset server

        Args:
            ds_name  : dataset file name
            scp_addr : url of dataset repository on remote server to scp from

        Returns:
            Boolean indicating whether file was found and downloaded

        """
        cmd = ["scp", os.path.join(scp_addr, ds_name), "data/"]

        try:
            logging.info("> %s" %(" ".join(cmd)))
            out   = subprocess.check_output(cmd)
            found = True
        except subprocess.CalledProcessError as e:
            found = False
            pass

        if (found):
            print("Downloaded from {}".format(scp_addr))
        else:
            print("Couldn't find dataset in {}".format(scp_addr))

        return found

    def run(self):
        """Execute download"""

        # Does the dataset exist on the local repository already?
        if (not os.path.exists(self.local_ds_path)):
            print("Dataset not available locally. Try to download from server.")

            for entry in self.cfg:
                addr = entry['user'] + "@" + entry['addr'] + ":" + entry['path']
                downloaded = self._download(self.ds_name, addr)
                if (downloaded):
                    return True
        else:
            logging.info("Dataset already available locally")


