"""Generate documentation for testbed dataset
"""
import glob
import json
import logging
import os
from datetime import timedelta

import json2html

import ptp.compression
from ptp.timestamping import Timestamp

logger = logging.getLogger(__name__)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


class Docs():
    def __init__(self, reset=False, cfg_path='data/'):
        if (cfg_path[0] == "/"):
            # Assume the target path is an absolute path
            self.cfg_path = os.path.abspath(cfg_path)
        else:
            # Assume the path is relative to this project's root directory
            this_file = os.path.realpath(__file__)
            rootdir = os.path.dirname(os.path.dirname(this_file))
            self.cfg_path = os.path.join(rootdir, "data")

        self.catalog_json = os.path.join(self.cfg_path, 'catalog.json')
        self.catalog_html = os.path.join(self.cfg_path, 'index.html')

        if (os.path.isfile(self.catalog_json) and reset):
            raw_resp = input(f"Catalog {self.catalog_json} already exists.\
                               Delete? [Y/n] ") or "Y"
            response = raw_resp.lower()

            if (response == 'n'):
                print("Aborting...")
                return
            else:
                os.remove(self.catalog_json)

        if (os.path.isfile(self.catalog_json)):
            with open(self.catalog_json) as fin:
                self.catalog = json.load(fin)
        else:
            self.catalog = list()

    def _read_metadata(self, filename):
        """Read metadata inside file passed as argument

        Args:
            filename: Path of the file

        Returns:
            Dictionary containing the metadata
        """
        codec = ptp.compression.Codec(filename=filename)
        dataset = codec.decompress()
        ds_name = os.path.basename(filename)
        ds_source = "testbed" if ds_name.split("-")[0] == "serial" else \
                    "simulation"

        if ('data' not in dataset or len(dataset['data']) == 0):
            logger.warning(f"File {filename} does not have any data")
            return

        # Check metadata for compatibility with old captures
        if ('metadata' in dataset and len(dataset['metadata']) > 0):
            metadata = dataset['metadata']
        else:
            metadata = {}

        # Add other relevant information to the metadata dictionary
        if (ds_source == "testbed"):
            t_end = Timestamp(dataset['data'][-1]["t2_sec"],
                              dataset['data'][-1]["t2"])
            t_start = Timestamp(dataset['data'][0]["t2_sec"],
                                dataset['data'][0]["t2"])
        else:
            t_end = dataset['data'][-1]["t2"]
            t_start = dataset['data'][0]["t2"]

        duration_ns = float(t_end - t_start)
        duration_tdelta = timedelta(microseconds=(duration_ns / 1e3))
        metadata["duration"] = str(duration_tdelta)
        metadata["size"] = sizeof_fmt(os.path.getsize(filename))
        metadata["n_exchanges"] = len(dataset['data'])
        metadata["source"] = ds_source.title()

        return metadata

    def add_dataset(self, file_path):
        """Add dataset metadata into dataset catalog

        Args:
            file_path : Path to dataset JSON file

        """
        try:
            metadata = self._read_metadata(file_path)
        except EOFError:
            logger.error(f"Dataset {file_path} missing termination")
            return  # probably a broken acquisition
        ds_name = os.path.basename(file_path)

        exists = False
        for entry in self.catalog:
            if entry['dataset'] == ds_name:
                logger.info(
                    f"Dataset {ds_name} already cataloged. Updating...")
                entry['info'] = metadata
                exists = True
                break

        if (not exists):
            self.catalog.append({'dataset': ds_name, 'info': metadata})

        sorted_catalog = sorted(self.catalog,
                                key=lambda k: k['dataset'],
                                reverse=True)

        with open(self.catalog_json, 'w') as fd:
            json.dump(sorted_catalog, fd, sort_keys=True, indent=2)

        logger.info(f"{ds_name} cataloged into {self.catalog_json}")

        # Re-generate .html catalog file
        json_data = json.dumps(sorted_catalog, sort_keys=True)
        html_head = """<!doctype html>
<html>
<head>
<!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css" integrity="sha384-HSMxcRTRxnN+Bdg0JdbxYKrThecOKuH5zCYotlSAcp1+c8xmyTe9GYg1l9a69psu" crossorigin="anonymous">

<!-- Optional theme -->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap-theme.min.css" integrity="sha384-6pzBo3FDv/PJ8r2KRkGHifhEocL+1X2rVCTTkUfGk7/0pbek5mMa1upzvWbrUbOZ" crossorigin="anonymous">

</head><body>
<!-- Latest compiled and minified JavaScript -->
<script src="https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js" integrity="sha384-aJ21OjlMXNL5UyIl/XNwTMqvzeRMZH2w8c5cRVpzpU8Y5bApTppSuUkhZXN0VxHd" crossorigin="anonymous"></script>
        """  # noqa: E501

        table_class = "class=\"table table-bordered table-hover\""
        html_body = json2html.json2html.convert(json=json_data,
                                                table_attributes=table_class)
        html_foot = "</body></html>"

        with open(self.catalog_html, 'w') as fd:
            fd.write(html_head)
            fd.write(html_body)
            fd.write(html_foot)

        logger.info(f"Updated HTML dataset catalog at {self.catalog_html}")

    def process(self):
        """Read all datasets of target directory and generate catalog"""

        # Add entry for each file
        extensions = [".json", ".pickle", ".gz", ".pbz2", ".xz"]
        all_datasets = list()
        for ext in extensions:
            all_datasets.extend(
                glob.glob(os.path.join(self.cfg_path, "**/*" + ext),
                          recursive=True))
        if (self.catalog_json in all_datasets):
            all_datasets.remove(self.catalog_json)

        # Catalog each dataset
        for dataset in sorted(all_datasets, reverse=True):
            print(f"Processing {dataset}")
            self.add_dataset(dataset)

        print("Saved {}".format(self.catalog_json))
        print("Saved {}".format(self.catalog_html))
