#!/usr/bin/env python

"""Acquisition of timestamps via UART
"""
import serial, time, json, logging, signal, os, shutil, subprocess
from pprint import pprint, pformat
from ptp.reader import Reader
from ptp.docs import Docs
import threading
from tabulate import tabulate
import pandas as pd
from collections import deque


logger = logging.getLogger(__name__)


class Serial():
    def __init__(self, rru_dev, bbu_dev, sensor_dev, n_samples, metadata):
        """Serial capture of timestamps from testbed

        Args:
            rru_dev    : RRU FPGA device ('rru_uart' or 'rru2_uart')
            bbu_dev    : BBU FPGA device ('bbu_uart')
            sensor_dev : Sensor device ('roe_sensor')
            n_samples  : Target number of samples (0 for infinity)
            metadata   : Information about the testbed configuration

        """
        self.n_samples = n_samples
        self.metadata  = metadata

        # Serial connections
        self.rru = self.connect(rru_dev)
        if (bbu_dev is not None):
            self.bbu  = self.connect(bbu_dev)
        else:
            self.bbu  = None
        if (sensor_dev is not None):
            self.sensor = self.connect(sensor_dev)
        else:
            self.sensor = None

        # Filename
        path = "data/"
        self.filename = path + "serial-" + time.strftime("%Y%m%d-%H%M%S") + ".json"

        # Initialize some vars
        self.last_temp    = (None, None)
        self.last_bbu_occ = None
        self.rru_data     = deque()

        # Continuously check that the devices are alive
        self.sensor_alive  = True
        self.bbu_alive     = True
        self.rru_alive     = True
        self.alive_timeout = 5 # in secs

        # Enable
        self.en_capture = True

        # Threads for reading the BBU and sensor (the RRU will be read by the
        # main thread)
        if (self.sensor is not None):
            sensor_thread = threading.Thread(target=self.read_sensor,
                                             daemon=True)
            sensor_thread.start()

        if (self.bbu is not None):
            bbu_thread = threading.Thread(target=self.read_bbu, daemon=True)
            bbu_thread.start()

        rru_thread = threading.Thread(target=self.read_rru, daemon=True)
        rru_thread.start()

    def _readline(self, dev):
        """Readline and clean whitespaces"""
        line = dev.readline().strip().decode("utf-8", "ignore")
        return " ".join(line.split())

    def _split_strip_line(self, line, key):
        """Strip all elements preceding the key element and split"""
        strip_line = line[line.find(key):]
        return strip_line.split()

    def read_sensor(self):
        """Loop for reading the sensor device"""
        last_read = time.time()

        while (self.en_capture):
            assert(self.sensor.in_waiting < 2048), \
                "Sensor serial buffer is getting full"

            temperature_str = self._readline(self.sensor)

            if (len(temperature_str) > 0):
                try:
                    temp_measurements = temperature_str.split(",")
                    self.last_temp = (float(temp_measurements[0]),
                                      float(temp_measurements[1]))
                except ValueError:
                        pass
                last_read = time.time()
            elif (time.time() - last_read > self.alive_timeout):
                self.sensor_alive = False
                logging.warning("Sensor device is unresponsive")
                break

    def read_bbu(self):
        """Loop for reading the BBU device"""
        last_read = time.time()

        while (self.en_capture):
            assert(self.bbu.in_waiting < 2048), \
                "BBU serial buffer is getting full"

            bbu_str = self._readline(self.bbu)

            if "Occupancy" in bbu_str:
                bbu_str_split = bbu_str.split(" ")
                if (len(bbu_str_split) >= 4):
                    try:
                        self.last_bbu_occ = int(bbu_str_split[3])
                    except ValueError:
                        pass

                last_read = time.time()
            elif (time.time() - last_read > self.alive_timeout):
                self.bbu_alive = False
                logging.warning("BBU is unresponsive")
                break

    def read_rru(self):
        """Loop for reading the RRU device"""
        last_read = time.time()

        rru_occ = None
        pps_err = None
        idx     = 0

        while (self.en_capture):
            assert(self.rru.in_waiting < 2048), \
                "RRU serial buffer is getting full"

            line = self._readline(self.rru)

            if (len(line) > 0):
                last_read = time.time()
            elif (time.time() - last_read > self.alive_timeout):
                self.rru_alive = False
                logging.warning("RRU is unresponsive")
                break

            # PPS time alignment error
            if '[pps-rtc][' in line:
                line_val = self._split_strip_line(line, "[pps-rtc][")
                if (line_val[1] == "Sync" and line_val[2] == "Error]"):
                    pps_err = int(line_val[3]) + float(line_val[5])/(2**32)

            # RRU occupancy
            if "Occupancy" in line:
                line_val = self._split_strip_line(line, "Occupancy:")
                if (len(line_val) >= 4):
                    try:
                        rru_occ = int(line_val[3])
                    except ValueError:
                        pass

            # PTP Timestamps
            if "Timestamps" in line:
                line_val = self._split_strip_line(line,  "Timestamps")

                # Normal PTP Timestamps
                seq_id = int(line_val[2])
                t1_ns  = int(line_val[4],16)
                t2_ns  = int(line_val[6],16)
                t3_ns  = int(line_val[8],16)
                t4_ns  = int(line_val[10],16)
                t1_sec = int(line_val[12],16)
                t2_sec = int(line_val[14],16)
                t3_sec = int(line_val[16],16)
                t4_sec = int(line_val[18],16)

                # PPS Timestamps
                t1_pps_ns  = int(line_val[20],16)
                t1_pps_sec = int(line_val[22],16)
                t4_pps_ns  = int(line_val[24],16)
                t4_pps_sec = int(line_val[26],16)

                # Append to results
                run_data = {
                    "idx"        : idx,
                    "seq_id"     : seq_id,
                    't1'         : t1_ns,
                    't2'         : t2_ns,
                    't3'         : t3_ns,
                    't4'         : t4_ns,
                    't1_sec'     : t1_sec,
                    't2_sec'     : t2_sec,
                    't3_sec'     : t3_sec,
                    't4_sec'     : t4_sec,
                    't1_pps'     : t1_pps_ns,
                    't1_pps_sec' : t1_pps_sec,
                    't4_pps'     : t4_pps_ns,
                    't4_pps_sec' : t4_pps_sec
                }
            else:
                continue

            # Append the last occupancy when available
            if (rru_occ is not None):
                run_data["rru_occ"] = rru_occ
                rru_occ = None

            # Append the last PPS error when available
            if (pps_err is not None):
                run_data["pps_err"] = pps_err
                pps_err = None

            self.rru_data.append(run_data)

            idx += 1

    def connect(self, device, baudrate=115200):
        """Establish a serial connection to a given device.

        Args:
            device : Target UART device within /dev

        Returns:
            Object with serial connection.

        """

        devices_list = ['bbu_uart',
                        'rru_uart',
                        'rru2_uart',
                        'roe_sensor']

        assert(device in devices_list), "Unknown UART device"

        dev_path = '/dev/' + device

        # Check whether device is busy and ask to kill process
        try:
            fuser_res = subprocess.check_output(["fuser", "-n", "file",
                                                 dev_path])
            resp = input("Process %d is reading from this device \
            - kill it? [Y/n] " %(int(fuser_res))) or "Y"
            if (resp.lower() == "y"):
                subprocess.run(["kill", fuser_res])
        except subprocess.CalledProcessError:
            # non-zero return code is when there is no process reading
            # from device
            pass

        serial_conn = serial.Serial(dev_path,
                                    baudrate = baudrate,
                                    bytesize = serial.EIGHTBITS,
                                    parity   = serial.PARITY_NONE,
                                    stopbits = serial.STOPBITS_ONE,
                                    timeout  = 1)
        logger.info("Connected to %s" %(device))
        return serial_conn

    def start_json_file(self):
        """Start the JSON file structure

        The JSON file is organized as a dictionary and contain the data and
        metadata. First, the file is initialized with the initial dict structure
        and the metadata information. After, list of dictionaries containing the
        testbed timestamps are saved to compose the data.

        """

        with open(self.filename, 'a') as fd:
            fd.write('{"metadata": ')
            json.dump(self.metadata, fd)
            fd.write(', "data":[')

    def end_json_file(self):
        """End the JSON file structure"""

        with open(self.filename, 'a') as fd:
            fd.write(']}')

    def save(self, data):
        """Save runner data on JSON file"""

        with open(self.filename, 'a') as fd:
            if (data['idx'] > 0):
                fd.write(',\n')
            json.dump(data, fd)

    def move(self):
        """Move JSON file"""
        dst_dir  = "/opt/ptp_datasets/"
        dst      = dst_dir + os.path.basename(self.filename)
        raw_resp = input(f"Move {self.filename} to {dst}? [Y/n] ") or "Y"
        response = raw_resp.lower()

        # Move dataset to '/opt/ptp_datasets/' and add entry on the dataset
        # catalog at '/opt/ptp_datasets/README.md'
        if (response == 'y'):
            # Move
            shutil.move(self.filename, dst)
            # Add to catalog
            docs = Docs(cfg_path=dst_dir)
            docs.add_dataset(dst) # assume file has already moved to dst

    def catch(self, signum, frame):
        self.en_capture = False
        self.end_json_file()
        logger.info("Terminating acquisition of %s" %(self.filename))
        self.move()
        logging.info("Run:\n./download.py %s" %(
            os.path.basename(self.filename)))
        exit()

    def run(self):
        """Continuously read from the RRU and collect timestamps
        """
        signal.signal(signal.SIGINT, self.catch)
        signal.siginterrupt(signal.SIGINT, False)

        # Use the reader class to post-process each set of timestamp in
        # real-time and to print the associated PTP metrics
        reader = Reader(infer_secs=False)

        self.start_json_file()

        logger.info("Starting capture")
        rru_occ      = None
        pps_err      = None
        last_seq_id  = None
        debug_buffer = list()

        last_read = time.time()
        count     = 0

        while self.en_capture == True and \
              ((count < self.n_samples) or self.n_samples == 0):

            # If a device becomes unresponsive, stop
            if ((not self.rru_alive) or
                (not self.bbu_alive) or
                (not self.sensor_alive)):
                logging.info("Unresponsive devices - stopping");
                break

            if (len(self.rru_data) > 0):
                run_data = self.rru_data.popleft()

                # Process PTP metrics for debugging
                reader.process(run_data, pr_level=logging.INFO)

                # Append the temperature and BBU occupancy when available
                if (self.last_temp[0] is not None or
                    self.last_temp[1] is not None):
                    run_data["temp"] = self.last_temp

                if (self.last_bbu_occ is not None):
                    run_data["bbu_occ"] = self.last_bbu_occ

                if ((last_seq_id is not None) and
                    (run_data['seq_id'] != ((last_seq_id + 1) % 2**16))):
                    logging.warning("PTP sequence id gap: {:d} to {:d}".format(
                        last_seq_id,
                        run_data['seq_id']
                    ))
                last_seq_id = run_data['seq_id']

                if (logger.root.level == logging.DEBUG):
                    debug_buffer.append(run_data)

                    if (run_data["idx"] % 20 == 19):
                        df = pd.DataFrame(debug_buffer)
                        print(tabulate(df, headers='keys', tablefmt='psql'))
                        debug_buffer.clear()

                # Append to output file
                self.save(run_data)
                count += 1
            else:
                time.sleep(0.1)

        self.end_json_file()
        self.move()


