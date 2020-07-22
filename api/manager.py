import os, sys, glob, logging, collections, datetime
sys.path.append('..')
from json.decoder import JSONDecodeError
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import InvalidRequestError, IntegrityError, SQLAlchemyError
import ptp.compression
import models


class Manager:
    def __init__(self, app, db=None, ds_folder=None):
        """Manage database operations

        Args:
            app       : Flask app object
            db        : SQLAlchemy object
            ds_folder : Dataset folder

        """
        self.app       = app
        self.db        = db
        self.model     = models.Dataset()
        self.logger    = app.logger
        self.ds_folder = ds_folder

    def _read_metadata(self, ds_path):
        """Decode/Load the dataset metadata

        Args:
            ds_path: Dataset path

        """
        metadata = None

        try:
            codec = ptp.compression.Codec(filename=ds_path)
            ds    = codec.decompress()
        except JSONDecodeError:
            self.logger.warning(f"Error while reading {ds_path}")
            return

        # Check metadata for compatibility with old captures
        if ('metadata' and 'data' in ds and len(ds['data']) > 0):
            metadata = ds['metadata']

            # Add additional information
            metadata["n_exchanges"] = len(ds['data'])
        else:
            if ('metadata' not in ds):
                self.logger.warning(f"File {ds_path} does not have metadata")
            if ('data' not in ds or len(ds['data']) == 0):
                self.logger.warning(f"File {ds_path} does not have any data")

        return metadata

    def search(self, parameters):
        """Apply a query on the database based on the passed parameters

        Args:
            parameters: Dictionary with all the possible parameters

        """
        filtered_par = {k: v for k, v in parameters.items() if v is not None}

        with self.app.app_context():
            try:
                query = self.model.query.filter_by(**filtered_par).all()
                return query
            except InvalidRequestError:
                self.logger.warning("Search parameters error")
                return None

    def search_by_name(self, ds_name):
        """Query dataset by name

        Args:
            ds_name: Dataset name

        """
        with self.app.app_context():
            query = self.model.query.filter_by(name=ds_name).first()

        return query

    def insert(self, ds_name, metadata):
        """Insert new dataset into the database

        Args:
            ds_name  : Dataset name
            metadata : Dataset metadata

        """
        assert(isinstance(self.db, SQLAlchemy))
        self.logger.info(f"Saving {ds_name} into database")

        formatter    = Formatter(ds_name, metadata)
        formatted_md = formatter.format()
        ds_entry     = models.Dataset(**formatted_md)

        exists = self.search_by_name(ds_name)
        if (exists):
            self.logger.info(f"{ds_name} already exists on the database")
            return

        with self.app.app_context():
            try:
                ds_entry.save()
            except SQLAlchemyError:
                self.logger.warning(f"Invalid metadata from {ds_name}")

    def delete(self, ds_name):
        """Delete dataset from database

        Args:
            ds_name : Dataset name

        """
        assert(isinstance(self.db, SQLAlchemy))
        try:
            with self.app.app_context():
                self.model.query.filter_by(name=ds_name).delete()
                self.db.session.commit()
                self.logger.info(f"Deleted dataset {ds_name}")
        except SQLAlchemyError as e:
            self.logger.warning(e)
            self.logger.info(f"Unable to delete {ds_name}")

    def populate(self):
        """Populate database with the available datasets"""

        self.logger.info("Running dataset populate")

        all_ds = list()
        for ext in ("*.json", "*.xz"):
            all_ds.extend(glob.glob(os.path.join(self.ds_folder, ext)))

        for dataset in all_ds:
            ds_name     = os.path.splitext(os.path.basename(dataset))[0]
            ds_metadata = self._read_metadata(dataset)

            if (ds_metadata is not None):
                self.insert(ds_name, ds_metadata)


class Formatter():
    def __init__(self, ds_name, metadata):
        """Format dataset metadata into the database model

        Args:
            ds_name  : Dataset name
            metadata : Metadata information

        """
        self.ds_name  = ds_name
        self.metadata = metadata

    def _remove_last_inner_key(self, key):
        new_key = key.split('_')
        new_key = '_'.join(new_key[:-1])

        return new_key

    def _flatten(self, data, parent_key='', sep='_'):
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
                items.extend(self._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def format(self):
        """Format dataset metadata

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
            'ptp_unicast': None,
            'departure_ctrl': None,
            'departure_gap': None,
            'tstamp_latency_corr': {
                'bbu': {
                    'tx': 80,
                    'rx': 80,
                },
                'rru1': {
                    'tx': 80,
                    'rx': 80,
                },
                'rru2': {
                    'tx': 80,
                    'rx': 80
                }
            },
            'start_time': None,
            'n_exchanges': None
        }

        # Generate flatten dictionaries
        flatten_exp_md = self._flatten(expected_md_structure)
        flatten_ds_md  = self._flatten(self.metadata)

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
                    find_key = self._remove_last_inner_key(find_key)
                try:
                    flatten_exp_md[k] = flatten_ds_md[find_key]
                except:
                    pass

        # Add dataset name
        flatten_exp_md['name'] = self.ds_name

        # Reduce 'fh_traffic' prefix to 'fh' on the final dictionary
        fh_keys = []
        for k in flatten_exp_md.keys():
            if 'fh_traffic' in k:
                fh_keys.append(k)
        for k in fh_keys:
            flatten_exp_md[k.replace('fh_traffic', 'fh')] = flatten_exp_md.pop(k)

        # Add boolean value to indicate the presence or absence of the FH
        # traffic
        flatten_exp_md['fh_traffic'] = False if (not self.metadata['fh_traffic']) else True

        return flatten_exp_md


