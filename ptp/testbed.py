#!/usr/bin/env python

"""Acquisition of timestamps via UART
"""
import time, json, logging, signal, os, shutil, math
from tabulate import tabulate
import pandas as pd
from collections import deque
from ptp.reader import Reader
from ptp.docs import Docs
import ptp.compression
from ptp import util


logger = logging.getLogger(__name__)


class Acquisition():
    def __init__(self, n_samples, metadata, roe, yes):
        """Data acquisition from the PTP testbed

        Gather PTP timestamp and complimentary data from the FPGA-based testbed.

        Args:
            n_samples   : Target number of samples (0 for infinity)
            metadata    : Information about the testbed configuration
            roe         : RoE object
            yes         : Yes to prompting by default

        """
        self.n_samples = n_samples
        self.metadata  = metadata
        self._yes      = yes

        # RoE object
        self.roe = roe

        # Filename
        path           = "data/"
        basename       = path + "serial-" + time.strftime("%Y%m%d-%H%M%S")
        self.json_file = basename + ".json"
        self.xz_file   = basename + "-comp.xz" # compressed file

        # State
        self.json_ended   = False
        self.sample_count = 0

    def _start_json_file(self):
        """Start the JSON file structure

        The JSON file is organized as a dictionary and contain the data and
        metadata. First, the file is initialized with the initial dict structure
        and the metadata information. After, list of dictionaries containing the
        testbed timestamps are saved to compose the data.

        """
        with open(self.json_file, 'a') as fd:
            fd.write('{"metadata": ')
            json.dump(self.metadata, fd)
            fd.write(', "data":[')

    def _end_json_file(self):
        """End the JSON file structure"""
        if (self.json_ended):
            return

        self.json_ended = True

        logging.info(f"Finalize {self.json_file} file")

        with open(self.json_file, 'a') as fd:
            fd.write(']}')

    def _save_json(self, data):
        """Save data on JSON file"""

        with open(self.json_file, 'a') as fd:
            if (data['idx'] > 0):
                fd.write(',\n')
            json.dump(data, fd)

    def _save_to_bin(self):
        """Read JSON file and save it in binary"""

        codec = ptp.compression.Codec(filename=self.json_file)
        codec.compress()
        codec.dump(ext="xz")

    def _move(self):
        """Move JSON file"""

        dst_dir  = "/opt/ptp_datasets/"
        dst      = dst_dir + os.path.basename(self.xz_file)

        # Move dataset to '/opt/ptp_datasets/' and add entry on the dataset
        # catalog at '/opt/ptp_datasets/README.md'
        if (self._yes or
            util.ask_yes_or_no(f"Move {self.xz_file} to {dst_dir}?")):
            # Move
            shutil.move(self.xz_file, dst)
            # Add to catalog
            docs = Docs(cfg_path=dst_dir)
            docs.add_dataset(dst) # assume file has already moved to dst

    def _catch(self, signum, frame):
        """Signal handler"""
        self._terminate(True)

    def _terminate(self, signal=False):
        """Terminate the acquisition"""
        self.roe.serial.stop()
        logger.info("Terminating acquisition of dataset")
        self._end_json_file()
        self._save_to_bin()

        if ((self.n_samples != 0) and (self.sample_count != self.n_samples)):
            logger.warning("Failed to acquire the target number of samples.")
            logger.warning("Target was {}, but acquired only {}".format(
                self.n_samples, self.sample_count))
            # In non-interactive mode, consider this as a failure and don't move
            # the dataset to the dataset repository
            if (self._yes):
                logger.warning("Dataset was not moved to repository")
                exit(-1)

            if (not util.ask_yes_or_no("Move dataset to repository anyway?")):
                exit()
        elif (self.n_samples == 0):
            # When acquiring infinite samples in non-interactive mode, if the
            # termination is being called by a signal handler, don't move the
            # dataset to the repository.
            if (self._yes and signal):
                exit(-1)

        self._move()
        logging.info("Run:\n./analyze.py -vvvv -f %s" %(
            os.path.basename(self.xz_file)))
        exit()

    def run(self):
        """Save/process timestamp sets and complementary data acquired serially
        """
        signal.signal(signal.SIGINT, self._catch)
        signal.signal(signal.SIGTERM, self._catch)

        # Run acquisition on RoE devices
        self.roe.serial.start(target="ts")

        # Use the reader class to post-process each set of timestamp in
        # real-time and to print the associated PTP metrics for debugging
        reader = Reader()

        # Prepare the JSON file
        self._start_json_file()

        logger.info("Starting capture")
        debug_buffer   = list()
        target_samples = math.inf if (self.n_samples == 0) else self.n_samples
        while (self.roe.serial.enabled and
               (self.sample_count < target_samples)):
            # Is there any data available?
            if (not self.roe.serial.data):
                time.sleep(0.1)
                continue

            data = self.roe.serial.data.popleft()

            # Process PTP metrics for debugging
            reader.process(data, pr_level=logging.INFO)

            if (logger.root.level == logging.DEBUG):
                debug_buffer.append(data)

                if (data["idx"] % 20 == 19):
                    df = pd.DataFrame(debug_buffer)
                    print(tabulate(df, headers='keys', tablefmt='psql'))
                    debug_buffer.clear()

            # Append to output JSON file
            self._save_json(data)
            self.sample_count += 1

        self._terminate()

