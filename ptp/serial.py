#!/usr/bin/env python

"""Acquisition of timestamps via UART
"""
import serial, time, json, logging, signal, os, shutil, subprocess
from ptp.reader import Reader
from ptp.docs import Docs
from ptp.roe import RoE
import threading
from tabulate import tabulate
import pandas as pd
from collections import deque


logger = logging.getLogger(__name__)


class Serial():
    def __init__(self, rru_dev, rru2_dev, bbu_dev, sensor_dev, n_samples,
                 metadata, roe_config):
        """Serial capture of timestamps from testbed

        Args:
            rru_dev    : RRU FPGA device ('rru_uart' or 'rru2_uart')
            rru2_dev   : RRU2 FPGA device ('rru_uart' or 'rru2_uart')
            bbu_dev    : BBU FPGA device ('bbu_uart')
            sensor_dev : Sensor device ('roe_sensor')
            n_samples  : Target number of samples (0 for infinity)
            metadata   : Information about the testbed configuration
            roe_config : RoE configuration data

        """
        self.n_samples = n_samples
        self.metadata  = metadata

        # RoE information and configuration data
        self.roe_config = roe_config

        # Serial connections
        assert(rru_dev != rru2_dev), "RRU and RRU2 devices should be different"
        self.rru    = self.connect(rru_dev)
        self.rru2   = None if (rru2_dev is None) else self.connect(rru2_dev)
        self.bbu    = None if (bbu_dev is None) else self.connect(bbu_dev)
        self.sensor = None if (sensor_dev is None) else self.connect(sensor_dev)

        # Initialize RoE manager object
        self.roe = RoE(self.metadata, self.roe_config,
                       self.rru,
                       self.rru2,
                       self.bbu)

        # Filename
        path = "data/"
        self.filename = path + "serial-" + time.strftime("%Y%m%d-%H%M%S") + ".json"

        # Timestamp sets read from the RRU:
        self.ts_data = deque()

        # Complementary data that is asynchronous to timestamp sets
        self.async_data = {
            "bbu_occ"  : deque(),
            "rru_occ"  : deque(),
            "rru2_occ" : deque(),
            "pps_err"  : deque(),
            "pps_err2" : deque(),
            "y_pps"    : deque(),
            "y_pps2"   : deque()
        }
        self.last_temp = (None, None)
        # NOTE: the temperature is asynchronous too, but it is faster than
        # timestamps logs. So we don't queue it. Instead, we get the last value.

        # Continuously check that the devices are alive
        self.sensor_alive  = True
        self.bbu_alive     = True
        self.rru_alive     = True
        self.rru2_alive    = True
        self.alive_timeout = 5 # in secs

        # Enable
        self.en_capture = True
        self.json_ended = False

        if (self.sensor is not None):
            sensor_thread = threading.Thread(target=self.read_sensor,
                                             daemon=True)
            sensor_thread.start()

        # Program and configure the RoE devices before starting the acquisition
        if (self.roe_config is not None and
            (self.roe_config['roe_prog'] or self.roe_config['roe_configure'])):
            self.roe.prog_and_configure()

        if (self.bbu is not None):
            bbu_thread = threading.Thread(target=self.read_bbu, daemon=True)
            bbu_thread.start()

        if (self.rru2 is not None):
            rru2_thread = threading.Thread(target=self.read_rru2, daemon=True)
            rru2_thread.start()

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

    def _read_occupancy(self, line, queue):
        """Read occupancy from log and save on queue"""

        if "Occupancy" in line:
            line_val = self._split_strip_line(line, "Occupancy:")
            if (len(line_val) >= 4):
                try:
                    queue.append(line_val[3])
                except ValueError:
                    pass

    def _read_pps_err(self, line, queue):
        """Read PPS sync error"""

        line_val = self._split_strip_line(line, "[pps-rtc][")
        if (line_val[1] == "Sync" and line_val[2] == "Error]"):
            pps_err = int(line_val[3]) + float(line_val[5])/(2**32)
            queue.append(pps_err)

    def _read_pps_pi_out(self, line, queue):
        """Read PPS sync PI loop output (frequency offset)

        NOTE: the PI loop used for PPS sync runs once a second. Hence, the PI
        controller/filter is expected to converge to the number of ns that are
        accumulated as time offset drift per second. This is equivalent to a
        frequency offset in ppb.

        """
        line_val = self._split_strip_line(line, "[pps-rtc][")
        if (line_val[1] == "PI" and line_val[2] == "Out]"):
            pi_out = int(line_val[3]) + float(line_val[5])/(2**32)
            queue.append(pi_out)

    def _read_timestamp_set(self, line, idx):
        """Read set of timestamps"""

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
        ts_data = {
            'idx'        : idx,
            'seq_id'     : seq_id,
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

        self.ts_data.append(ts_data)

    def read_sensor(self):
        """Loop for reading the sensor device"""

        last_read = time.time()

        while (self.en_capture):
            assert(self.sensor.in_waiting < 2048), \
                "Sensor serial buffer is getting full"

            temperature_str = self._readline(self.sensor)

            if (len(temperature_str) > 1):
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

            line = self._readline(self.bbu)

            if (len(line) > 0):
                last_read = time.time()
            elif (time.time() - last_read > self.alive_timeout):
                self.bbu_alive = False
                logging.warning("BBU is unresponsive")
                break

            if "Occupancy" in line:
                self._read_occupancy(line, self.async_data["bbu_occ"])

    def read_rru2(self):
        """Loop for reading the RRU2 device"""

        last_read = time.time()

        while (self.en_capture):
            assert(self.rru2.in_waiting < 2048), \
                "RRU2 serial buffer is getting full"

            line = self._readline(self.rru2)

            if (len(line) > 0):
                last_read = time.time()
            elif (time.time() - last_read > self.alive_timeout):
                self.rru2_alive = False
                logging.warning("RRU2 is unresponsive")
                break

            if '[pps-rtc][' in line:
                if "Sync Error" in line:
                    self._read_pps_err(line, self.async_data["pps_err2"])
                elif "PI Out" in line:
                    self._read_pps_pi_out(line, self.async_data["y_pps2"])

            if "Occupancy" in line:
                self._read_occupancy(line, self.async_data["rru2_occ"])

    def read_rru(self):
        """Loop for reading the RRU device"""

        last_read = time.time()
        idx = 0
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

            if '[pps-rtc][' in line:
                if "Sync Error" in line:
                    self._read_pps_err(line, self.async_data["pps_err"])
                elif "PI Out" in line:
                    self._read_pps_pi_out(line, self.async_data["y_pps"])

            if "Occupancy" in line:
                self._read_occupancy(line, self.async_data["rru_occ"])

            if "Timestamps" in line:
                self._read_timestamp_set(line, idx)
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
        if (self.json_ended):
            return

        self.json_ended = True

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
        """Save/process timestamp sets and complementary data acquired serially
        """
        signal.signal(signal.SIGINT, self.catch)
        signal.siginterrupt(signal.SIGINT, False)

        # Use the reader class to post-process each set of timestamp in
        # real-time and to print the associated PTP metrics for debugging
        reader = Reader()

        self.start_json_file()

        logger.info("Starting capture")
        last_seq_id  = None
        debug_buffer = list()
        count        = 0
        while self.en_capture == True and \
              ((count < self.n_samples) or self.n_samples == 0):

            # If a device becomes unresponsive, stop
            if ((not self.rru_alive) or
                (not self.bbu_alive) or
                (not self.sensor_alive)):
                logging.info("Unresponsive devices - stopping");
                break

            if (self.ts_data):
                run_data = self.ts_data.popleft()

                # Process PTP metrics for debugging
                reader.process(run_data, pr_level=logging.INFO)

                # Latest temperature reading
                if (self.last_temp[0] is not None or
                    self.last_temp[1] is not None):
                    run_data["temp"] = self.last_temp

                # Complementary/asynchronous data
                for key in self.async_data:
                    if (self.async_data[key]):
                        run_data[key] = self.async_data[key].popleft()

                # Sanity check on PTP sequenceId
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


