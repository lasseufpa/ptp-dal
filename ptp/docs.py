"""Generate documentation for testbed dataset
"""
import logging, os, json, glob
import markdown2

logger = logging.getLogger(__name__)


class Docs():
    def __init__(self, cfg_path='/opt/ptp_datasets/'):

        self.header    = None
        self.cfg_path  = os.path.abspath(cfg_path)
        self.cfg_file  = os.path.join(cfg_path, 'README.md')
        self.html_file = os.path.join(cfg_path, 'README.html')
        self.values    = list()

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
                          'FH Traffic UL': 'ul', \
                          'Experiment': 'experiment'}

        for h in self.header.keys():
            header_line    += '|' + h
            separator_line += '|' + '-'*len(h)

        with open(self.cfg_file, 'w') as fd:
            fd.write(header_line + '|\n')
            fd.write(separator_line + '|\n')

    def add_value(self, filename):
        """Add row with metadata to markdown table

        Args:
            filename : Dataset path
            metadata : Dictionary with metadata

        """
        assert(os.path.exists(self.cfg_file)), "README.md file not exist"

        values     = list()
        value_line = ''
        metadata   = self.read_metadata(filename)

        if (metadata is None):
            return

        for h in self.header.values():
            if (h == 'file'):
                values.append(os.path.basename(filename))
            # Some datasets are part of an experiment involving other
            # datasets. In this case, the convention is that all datasets of the
            # the experiment will be contained on a subdirectory of the
            # top-level dataset directory/repository.
            elif (h == 'experiment'):
                dataset_dir = os.path.dirname(os.path.abspath(filename))
                experiment  = dataset_dir.replace(self.cfg_path, '')
                if (not experiment):
                    values.append('None')
                else:
                    values.append(experiment[1:]) # skip the first slash
            else:
                v = self._find_value(metadata, h)
                if (v):
                    values.append(v)
                else:
                    values.append('None')

        for v in values:
            if (v is None):
                v = ''

            value_line += '|' + str(v)

        with open(self.cfg_file, 'a') as fd:
            fd.write(value_line + '|\n')

    def _find_value(self, obj, key):
        """Find value corresponding to key in a nested dictionary

        Args:
            obj: Dictionary to search
            key: Key of the wanted value

        """
        if key in obj:
            return obj[key]

        for k,v in obj.items():
            if (isinstance(v,dict) and (v is not None)):
                nested_find = self._find_value(v, key)
                if (nested_find):
                    return nested_find

    def read_metadata(self, filename):
        """Read metadata inside file passed as argument

        Args:
            filename: Path of the file

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
                print("Aborting...")
                return

        # Generate the header
        self.set_header()

        # Add row for each file
        all_datasets = glob.glob(os.path.join(self.cfg_path, "**/*.json"),
                                 recursive=True)

        # Generate .md file
        for dataset in all_datasets:
            print(dataset)
            self.add_value(dataset)

        # Compile .md file to HTML
        html = markdown2.markdown_path(self.cfg_file,
                                       extras=["tables", "code-friendly"])
        with open(self.html_file, 'w') as fd:
            fd.write(html)

