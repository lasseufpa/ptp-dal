#!/usr/bin/env python

"""Generate documentation for testbed dataset
"""
import logging, os, json

logger = logging.getLogger(__name__)


class Docs():
    def __init__(self, cfg_path='/opt/ptp_datasets/'):

        self.header   = None
        self.cfg_path = cfg_path
        self.cfg_file = cfg_path + 'README.md'
        self.values   = list()

    def set_header(self):
        """Generate header of the markdown table"""

        separator_line = ''
        header_line    = ''
        self.header    = {'File': 'file', \
                          'Start Time': 'start_time', \
                          'Oscillator type': 'oscillator', \
                          'Sync Period': 'sync_period', \
                          'Number of hops': 'hops', \
                          'FH Traffic DL': 'dl', \
                          'FH Traffic UL': 'ul'}

        header        = [k for k,v in self.header.items()]

        for h in header:
            header_line    += '|' + h
            separator_line += '|' + '-'*len(h)

        with open(self.cfg_file, 'w') as fd:
            fd.write(header_line + '|\n')
            fd.write(separator_line + '|\n')

    def add_value(self, filename):
        """Add row with metadata to markdown table

        Args:
            filename:
            metadata:

        """
        value      = list()
        value_line = ''
        header     = [v for k,v in self.header.items()]
        metadata   = self.read_metadata(filename)

        for h in header:
            if (h == 'file'):
                value += [filename]
            else:
                value += [self._find_value(metadata, h)]

        for v in value:
            if (v is None):
                v = ''

            value_line += '|' + str(v)

        with open(self.cfg_file, 'a') as fd:
            fd.write(value_line + '|\n')

    def _find_value(self, obj, key):
        """Find value in a multilevel dictionary

        Args:
            obj: Dictionary to search
            key: Key of the wanted value

        """
        if key in obj:
            return obj[key]
        for k,v in obj.items():
            if (isinstance(v,dict) and (v is not None)):
                return self._find_value(v, key)
            else:
                return ''

    def read_metadata(self, filename):
        """Read metadata inside file passed as argument

        Args:
            filename:

        Returns:
            Dictionary containing the metadata
        """
        metadata = None
        if (filename.endswith('.json')):
            with open(filename) as fin:
                fd = json.load(fin)

            # Check metadata for compatibility with old captures
            if ('metadata' in fd):
                metadata = fd['metadata']
            else:
                logger.info(f"File {filename} has no metadata")

        return metadata

    def process(self):
        """Process all files on folder and generate a markdown table"""

        if (os.path.isfile(self.cfg_file)):
            raw_resp = input(f"Markdown table {self.cfg_file} exists.\
                               Delete? [Y/n] ") or "Y"
            response = raw_resp.lower()

            if (response == 'n'):
                return

        # Generate the header
        self.set_header()

        # Add row for each file
        basepath = os.path.dirname(self.cfg_path)
        for entry in os.listdir(basepath):
            filename = os.path.join(basepath, entry)
            if os.path.isfile(filename):
                metadata = self.read_metadata(filename)
                if (metadata is not None):
                    self.add_value(filename, metadata)

