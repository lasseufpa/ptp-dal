"""Dataset manager
"""
import json
import logging
import os
import shutil
import subprocess

import requests

from ptp import util, docs

logger = logging.getLogger(__name__)


class Datasets():
    def __init__(self):
        """Dataset manager"""
        self._set_paths()
        self._check_cfg()

        self.cfg = self._load_cfg()
        if (self.cfg is None):
            logger.error("Failed to load dataset server configurations")

        self.api_url = 'https://ptp.database.lasseufpa.org/api/'

    def _set_paths(self):
        """Define paths to save the configuration file"""
        this_file = os.path.realpath(__file__)
        rootdir = os.path.dirname(os.path.dirname(this_file))
        self.local_repo = os.path.join(rootdir, "data")
        home = os.path.expanduser("~")
        self.cfg_path = os.path.join(home, ".ptp")
        self.cfg_file = os.path.join(self.cfg_path, "config.json")

        # Create local repo if it does not exist
        if not os.path.isdir(self.local_repo):
            os.makedirs(self.local_repo)

    def _get_all_ds_variations(self, dataset):
        """Get all possible variations to the dataset name"""
        self.ds_name = os.path.basename(dataset)
        no_ext_ds_name = os.path.splitext(self.ds_name)[0]
        ds_prefix = no_ext_ds_name.replace("-comp", "")
        ds_suffixes = [
            "-comp.xz", "-comp.pbz2", "-comp.gz", "-comp.pickle", "-comp.json",
            ".json"
        ]
        all_ds_names = [ds_prefix + suffix for suffix in ds_suffixes]
        all_local_paths = [os.path.join(self.local_repo, d) for d in \
                           all_ds_names]

        return all_local_paths, all_ds_names

    def _check_cfg(self):
        """Check if path to cfg folder exists or create it otherwise"""
        if (not os.path.exists(self.cfg_path)):
            os.mkdir(self.cfg_path)
        elif (not os.path.isdir(self.cfg_path)):
            raise IsADirectoryError(
                "{} already exists, but is not a directory".format(
                    self.cfg_path))

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

    def _copy_key_cert(self, key, cert):
        """Copy key and digital certificate into the config directory"""
        certs_dir = os.path.join(self.cfg_path, "certs")

        # Create certs directory if it does not exist
        if not os.path.isdir(certs_dir):
            os.makedirs(certs_dir)

        # Copy key and cert to the certs directory
        new_key_path = shutil.copy(os.path.expanduser(key), certs_dir)
        new_crt_path = shutil.copy(os.path.expanduser(cert), certs_dir)

        return new_key_path, new_crt_path

    def _create_cfg(self):
        """Create configuration file with user-provided credentials"""
        if (not util.ask_yes_or_no("Provide information now?")):
            return

        cfg = list()
        more = True
        while (more):
            dl_mode = input("Download via API or SSH? (API) ") or "API"

            if (dl_mode.upper() == 'SSH'):
                server = input("IP address of the dataset server: ")
                path = input("Path to dataset repository on server: ")
                user = input("Username to access the server: ")
                cfg.append({
                    'dl_mode': 'SSH',
                    'addr': server,
                    'path': path,
                    'user': user
                })

            elif (dl_mode.upper() == 'API'):
                ssl_key_in = input("Path to your SSL key: ")
                ssl_crt_in = input("Path to your SSL certificate: ")
                ssl_key, ssl_crt = self._copy_key_cert(ssl_key_in, ssl_crt_in)
                cfg.append({
                    'dl_mode': 'API',
                    'ssl_key': ssl_key,
                    'ssl_crt': ssl_crt
                })

            else:
                raise ValueError(
                    "Download mode {} not defined".format(dl_mode))

            more = util.ask_yes_or_no("Add another address?")

        with open(self.cfg_file, 'w') as fd:
            json.dump(cfg, fd)

        logger.info(f"Saved dataset server configurations on {self.cfg_file}")
        return cfg

    def _download_ssh(self, cfg, ds_name):
        """Download dataset via SSH from dataset server

        Args:
            cfg     : Configuration file with user and server information
            ds_name : Dataset file name

        Return:
            Path to the file that was downloaded. None if not found.

        """
        ds_repo = cfg['user'] + "@" + cfg['addr'] + ":" + cfg['path']
        scp_src = os.path.join(ds_repo, ds_name)
        cmd = ["scp", scp_src, "data/"]
        ds_path = None

        logger.info("Trying %s" % (scp_src))
        res = subprocess.run(cmd,
                             timeout=60.0,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

        if (res.returncode == 0):
            print("Downloaded {} from {}".format(ds_name, ds_repo))
            ds_path = os.path.join("data", ds_name)
        else:
            logger.debug("Couldn't find file {} in {}".format(
                ds_name, ds_repo))

        return ds_path

    def _download_api(self, cfg, ds_name):
        """Download dataset via RESTful API

        Args:
            cfg     : Configuration file with user and server information
            ds_name : Dataset file name

        Return:
            Path to the file that was downloaded. None if not found.

        """
        addr = self.api_url + 'dataset'
        ds_req = os.path.join(addr, ds_name)
        logger.info("Trying " + ds_req)
        found = False
        ds_path = None

        try:
            cert = (cfg['ssl_crt'], cfg['ssl_key'])
            req = requests.get(ds_req, cert=cert, timeout=60.0)
            req.raise_for_status()
            local_ds_path = os.path.join(self.local_repo, ds_name)
            open(local_ds_path, 'wb').write(req.content)
            found = True
        except requests.exceptions.RequestException as e:
            pass

        if (found):
            print("Downloaded {} from {}".format(ds_name, addr))
            ds_path = local_ds_path
        else:
            logger.debug("Couldn't find file {} in {}".format(ds_name, addr))

        return ds_path

    def download(self, dataset):
        """Download dataset from dataset server

        Args:
            dataset : dataset name/path

        Returns:
            Path to dataset file

        """
        all_local_paths, all_ds_names = self._get_all_ds_variations(dataset)
        # Does the dataset exist on the local repository already?
        for path in all_local_paths:
            if (os.path.exists(path)):
                logger.info("Dataset already available locally")
                return path

        print("Dataset not available locally. Try to download from server.")

        # Try to load in order of compression (most compressed first)
        for entry in self.cfg:
            dl_mode = entry['dl_mode']
            for ds_name in all_ds_names:
                if dl_mode == 'SSH':
                    ds_path = self._download_ssh(entry, ds_name)
                else:
                    ds_path = self._download_api(entry, ds_name)

                if (ds_path is not None):
                    break

            if (ds_path is not None):
                break

        # Add to local catalog
        if (ds_path is not None):
            catalog = docs.Docs()
            catalog.add_dataset(ds_path)
            return ds_path

        # We should have returned before if a datset was found on the local or
        # remote repositories
        raise RuntimeError("Couldn't find dataset")

    def search(self, parameters):
        """Search datasets via RESTful API

        Args:
            parameters : Dictionary with the query parameters

        Return:
            List with the founded datasets

        """
        addr = self.api_url + 'search'
        headers = {'content-type': 'application/json'}
        ds_found = None

        if (self.cfg is None):
            return

        api_connections = [e for e in self.cfg if (e['dl_mode'] == 'API')]

        if (len(api_connections) == 0):
            logger.error(
                "Couldn't find a dataset server in your configuration")
            return

        for conn in api_connections:
            cert = (conn['ssl_crt'], conn['ssl_key'])
            try:
                req = requests.post(addr,
                                    data=json.dumps(parameters),
                                    headers=headers,
                                    cert=cert)
                req.raise_for_status()
                response = req.json()
                ds_found = response['found']
            except requests.exceptions.RequestException as e:
                if (req.status_code == 400):
                    logger.info("Bad request! Check your cfg file.")
                elif (req.status_code == 404):
                    logger.info("No dataset found!")
                else:
                    logger.info(e)
                pass

        return ds_found
