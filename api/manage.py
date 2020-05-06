import os, sys, glob, logging, collections, datetime
sys.path.append('..')
import ptp.compression
from models import Dataset
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class Database:
    """Manage database operations"""
    def __init__(self, app=None, logger=False):
        self.app    = app
        self.logger = app.logger if (logger) else None

    def remove_last_inner_key(self, key):
        new_key = key.split('_')
        new_key = '_'.join(new_key[:-1])

        return new_key

    def flatten(self, data, parent_key='', sep='_'):
        """Flatten deep dictionaries preserving the keys

        Example:
            data   = {'a': {'b': 2, 'c': 3}}
            output = {'a_b': 2, 'a_c': 3}

        Args:
            data : Deep dictionary to be flatten

        Output:
            Flatten dictionary

        """
        items = []
        for k, v in data.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def format_metadata(self, ds_name, metadata):
        """Format dataset metadata

        Args:
            ds_name : Dataset name
            md      : Metadata information

        Returns:
            Dictionary in the correct format to save in the database

        """
        expected_md_structure = {
            'oscillator': None,
            'sync_period': None,
            'fh_traffic': {
                'type': None,
                'fs': None,
                'bitrate': {
                    'dl': None,
                    'ul': None
                },
                'iq_size': {
                    'dl': None,
                    'ul': None
                },
                'n_spf': {
                    'dl': None,
                    'ul': None,
                },
                'n_rru_dl': None,
                'n_rru_ul': None,
                'vlan_pcp': None,
            },
            'delay_cal': None,
            'delay_cal_duration': None,
            'hops': {
                'rru1': None,
                'rru2': None,
            },
            'n_rru_ptp': None,
            'pipeline': {
                'bbu': None,
                'rru': None
            },
            'start_time': None
        }

        # Generate flatten dictionaries
        flatten_exp_md = self.flatten(expected_md_structure)
        flatten_ds_md  = self.flatten(metadata)

        # Find the matches between the expected dictionary and the actual one
        for k,v in flatten_exp_md.items():
            if ("pipeline_" in k):
                # Special case. For the pipeline, we get e.g. "pipelines_bbu",
                # but we want to save it as "pipeline_bbu".
                incoming_k = "pipelines_{}".format(k.split("_")[-1])
                if (incoming_k in flatten_ds_md):
                    flatten_exp_md[k] = flatten_ds_md[incoming_k]
            elif (k == "start_time"):
                flatten_exp_md[k] = datetime.datetime.strptime(
                    flatten_ds_md[k], '%Y-%m-%d %H:%M:%S')
            elif k in flatten_ds_md:
                flatten_exp_md[k] = flatten_ds_md[k]
            else:
                # Look if the key is expected to be a outer key or inner key.
                # If its a outer key and it is not found, assume None value
                if k in expected_md_structure.keys():
                    continue

                # If it is an inner key, try to find the corresponding key
                find_key = k
                while find_key not in flatten_ds_md:
                    # Remove last inner key until find some corresponding on the
                    # metadata
                    if not find_key:
                        # If the find_key is empty means that the key is not in
                        # the metadata
                        break
                    # Remove the last key and search in the dictionary again
                    find_key = self.remove_last_inner_key(find_key)
                try:
                    flatten_exp_md[k] = flatten_ds_md[find_key]
                except:
                    pass

        # Add dataset name
        flatten_exp_md['name'] = ds_name

        # Reduce keynames on the final dictionary
        red_keys = {'fh_traffic' : 'fh'}
        for old_key,new_key in red_keys.items():
            for k in flatten_exp_md.keys():
                flatten_exp_md[k.replace(old_key, new_key)] = flatten_exp_md.pop(k)

        # Add boolean value to indicate the presence or absence of the FH
        # traffic
        flatten_exp_md['fh_traffic'] = False if (not metadata['fh_traffic']) else True

        return flatten_exp_md

    def drop(self):
        """Drops database tables"""
        with self.app.app_context():
            Dataset.drop_all()

    def create(self):
        """Creates database tables from sqlalchemy models"""
        with self.app.app_context():
            Dataset.create_all()

    def populate(self):
        """Populate database with the available datasets"""

        DS_PATH = os.getenv('DS_PATH')
        assert(DS_PATH), "Unable to find dataset path"

        all_ds = list()
        for ext in ("*.json", "*.xz"):
            all_ds.extend(glob.glob(os.path.join(DS_PATH, ext)))

        for dataset in all_ds:
            with self.app.app_context():
                ds_name = os.path.splitext(os.path.basename(dataset))[0]
                query   = Dataset.search({'name': ds_name})

            if (query):
                print(f"Already saved on the database: {query}")
                continue

            try:
                # Load and decompress dataset
                codec = ptp.compression.Codec(filename=dataset)
                ds    = codec.decompress()
            except:
                print(f"Error while reading {ds_name}")
                continue

            # Check metadata
            if ("metadata" in ds):
                metadata = ds['metadata']
            else:
                self.logger.warning(f"Unable to find metadata information in {ds_name}")
                continue

            try:
                # Create dataset model and save
                self.logger.info(f"Saving {ds_name} into database")
                ds_metadata = self.format_metadata(ds_name, metadata)
                ds_entry    = Dataset(**ds_metadata)
                with self.app.app_context():
                    ds_entry.save()
            except:
                self.logger.warning(f"Invalid metadata information for ds {ds_name}")


class Watcher:
    def __init__(self, app):
        self.path     = os.getenv('DS_PATH')
        self.handler  = Handler(self.path, app)
        self.observer = Observer()
        self.db       = Database(app, logger=True)

        # Init the database on the first run
        self.db.populate()

    def run(self):
        self.observer.schedule(self.handler, self.path, recursive=True)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()


class Handler(PatternMatchingEventHandler):
    def __init__(self, path, app):
        """Handler constructure"""
        super(Handler, self).__init__(patterns=['*.json', '*.xz'],
                                      ignore_patterns=["*~"],
                                      ignore_directories=True,
                                      case_sensitive=True)
        self.path   = path
        self.db     = Database(app, logger=True)
        self.logger = app.logger

    def on_moved(self, event):
        pass

    def on_created(self, event):
        self.logger.info("Running database populate")
        self.db.populate()

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass
