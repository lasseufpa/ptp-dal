#!/usr/bin/env python

"""Acquisition of timestamps via UART
"""
import serial, time, json, logging, signal, os, shutil, subprocess
from ptp.reader import Reader
from ptp.docs import Docs
from pyroe.roe import roe
import threading
from tabulate import tabulate
import pandas as pd
from collections import deque
import ptp.compression


logger = logging.getLogger(__name__)


class Serial():
    def __init__(self, rru_dev, rru2_dev, bbu_dev, sensor_dev, n_samples,
                 metadata, roe_config, baudrate, yes=False):
        """Serial capture of timestamps from testbed

        Args:
            rru_dev    : RRU FPGA device ('rru_uart' or 'rru2_uart')
            rru2_dev   : RRU2 FPGA device ('rru_uart' or 'rru2_uart')
            bbu_dev    : BBU FPGA device ('bbu_uart')
            sensor_dev : Sensor device ('roe_sensor')
            n_samples  : Target number of samples (0 for infinity)
            metadata   : Information about the testbed configuration
            roe_config : RoE configuration data
            yes        : Yes to prompting by default

        """
        self.n_samples = n_samples
        self.metadata  = metadata

        # Check if there is another acquisition running
        self._assert_free_to_capture()

        # Check if submodules are up-to-date
        self._check_git_submodules()

        # RoE information and configuration data
        self.roe_config = roe_config

        # Serial connections
        assert(rru_dev != rru2_dev), "RRU and RRU2 devices should be different"
        self._yes   = yes
        self.rru    = self.connect(rru_dev, baudrate=baudrate)
        self.rru2   = None if (rru2_dev is None) else \
                      self.connect(rru2_dev, baudrate=baudrate)
        self.bbu    = None if (bbu_dev is None) else \
                      self.connect(bbu_dev, baudrate=baudrate)
        self.sensor = None if (sensor_dev is None) else \
                      self.connect(sensor_dev, baudrate=115200)

        # Configure RoE network
        hops      = "{},{}".format(metadata["hops"]["rru1"],
                                   metadata["hops"]["rru2"])
        set_cmd   = ["python3", "switch_ctrl.py", "--json", "set", hops]
        read_cmd  = ["python3", "switch_ctrl.py", "--json", "read"]
        subprocess.run(set_cmd, cwd="roe-instruments")
        net_cfg   = subprocess.check_output(read_cmd, cwd="roe-instruments")
        print(net_cfg.decode())

        # Initialize RoE manager object
        self.roe = roe.RoE(self.metadata, self.roe_config, self.rru, self.rru2,
                           self.bbu)

        # Filename
        path = "data/"
        basename         = path + "serial-" + time.strftime("%Y%m%d-%H%M%S")
        self.json_file   = basename + ".json"
        self.xz_file     = basename + "-comp.xz" # compressed file

        # Timestamp sets read from the RRU:
        self.ts_data     = deque()
        self.ts_last_sec = None

        # Complementary data that is asynchronous to timestamp sets
        self.async_data = {
            "bbu_occ"  : deque(),
            "rru_occ"  : deque(),
            "rru2_occ" : deque(),
            "pps_err"  : deque(),
            "pps_err2" : deque(),
            "y_pps"    : deque(),
            "y_pps2"   : deque(),
            "temp"     : deque()
        }

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

        if (self.rru2 is not None and self.roe.rru2.active):
            rru2_thread = threading.Thread(target=self.read_rru2, daemon=True)
            rru2_thread.start()

        rru_thread = threading.Thread(target=self.read_rru, daemon=True)
        rru_thread.start()

    def _assert_free_to_capture(self):
        """Verify whether we are free to capture from FPGAs

        Throw exception in case there is another acquisition running.

        """
        our_pid = os.getpid()
        res     = subprocess.check_output(["pgrep", "-f", "capture.py", "-a"])

        for line in res.splitlines():
            pid = int(line.decode().split()[0])
            if (pid != our_pid):
                raise RuntimeError("An acquisition is already running on PID "
                                   f"{pid}")

    def _check_git_submodules(self):
        """Check if Git submodules are up-to-date"""

        res = subprocess.check_output(["git", "submodule", "status"])

        outdated = False
        for line in res.decode().splitlines():
            if ("+" in line):
                submodule = line.split()[1]
                logger.warning(f"{submodule} is not up-to-date")
                outdated = True

        if (outdated):
            raise RuntimeError("Git submodules are not up-to-date. "
                               "Run \"git submodule update\"")

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
                    queue.append(int(line_val[3]))
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

        line_val = self._split_strip_line(line,  "Ts:")

        # The RoE device will either print the full set with seconds and
        # nanoseconds of six timestamps, or print a reduced set containing only
        # the nanoseconds. The device prints the reduced set when the values of
        # seconds of all timestamps coincide with a previously printed value. In
        # this case, the receive end (here) can infer the values of seconds.
        if (len(line_val) == 14):
            # Normal PTP Timestamps
            seq_id     = int(line_val[1],16)
            t1_ns      = int(line_val[2],16)
            t2_ns      = int(line_val[3],16)
            t3_ns      = int(line_val[4],16)
            t4_ns      = int(line_val[5],16)
            t1_sec     = int(line_val[6],16)
            t2_sec     = int(line_val[7],16)
            t3_sec     = int(line_val[8],16)
            t4_sec     = int(line_val[9],16)
            # PPS Timestamps
            t1_pps_ns  = int(line_val[10],16)
            t1_pps_sec = int(line_val[11],16)
            t4_pps_ns  = int(line_val[12],16)
            t4_pps_sec = int(line_val[13],16)
            # The RoE hardware takes t1_sec as reference for the last printed
            # value of seconds
            self.ts_last_sec = t1_sec
        elif (len(line_val) == 8):
            if (self.ts_last_sec is None):
                raise ValueError("Can't infer the values of seconds")
            # Normal PTP Timestamps
            seq_id     = int(line_val[1],16)
            t1_ns      = int(line_val[2],16)
            t2_ns      = int(line_val[3],16)
            t3_ns      = int(line_val[4],16)
            t4_ns      = int(line_val[5],16)
            # PPS Timestamps
            t1_pps_ns  = int(line_val[6],16)
            t4_pps_ns  = int(line_val[7],16)
            # Repeated seconds
            t1_sec     = self.ts_last_sec
            t2_sec     = self.ts_last_sec
            t3_sec     = self.ts_last_sec
            t4_sec     = self.ts_last_sec
            t1_pps_sec = self.ts_last_sec
            t4_pps_sec = self.ts_last_sec
        else:
            raise ValueError("Missing elements on timestamp set")


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
                    temp_measurements_str = temperature_str.split(",")
                    temp_measurements = (float(temp_measurements_str[0]),
                                         float(temp_measurements_str[1]))
                    self.async_data["temp"].append(temp_measurements)
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

            if (self.roe.bbu.wait_free()):
                # Reset once the mutex is released since the wait could have
                # lasted a long time
                self.bbu.reset_input_buffer()

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

            if (self.roe.rru2.wait_free()):
                # Reset once the mutex is released since the wait could have
                # lasted a long time
                self.rru2.reset_input_buffer()

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

            if (self.roe.rru.wait_free()):
                # Reset once the mutex is released since the wait could have
                # lasted a long time
                self.rru.reset_input_buffer()

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

            if "Ts:" in line:
                try:
                    self._read_timestamp_set(line, idx)
                    idx += 1
                except ValueError:
                    logging.warning("Failed to parse timestamp set for "
                                    "line:\n{}".format(line))

    def connect(self, device, baudrate):
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
            if (not self._yes):
                resp = input("Process %d is reading from this device \
                - kill it? [Y/n] " %(int(fuser_res))) or "Y"
            if (self._yes or (resp.lower() == "y")):
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
        with open(self.json_file, 'a') as fd:
            fd.write('{"metadata": ')
            json.dump(self.metadata, fd)
            fd.write(', "data":[')

    def end_json_file(self):
        """End the JSON file structure"""
        if (self.json_ended):
            return

        self.json_ended = True

        logging.info(f"Finalize {self.json_file} file")

        with open(self.json_file, 'a') as fd:
            fd.write(']}')

    def save_json(self, data):
        """Save data on JSON file"""

        with open(self.json_file, 'a') as fd:
            if (data['idx'] > 0):
                fd.write(',\n')
            json.dump(data, fd)

    def save_to_bin(self):
        """Read JSON file and save it in binary

        @note: Protocol version 4 was added in Python 3.4.

        """
        codec = ptp.compression.Codec(filename=self.json_file)
        codec.compress()
        codec.dump(ext="xz")

    def move(self):
        """Move JSON file"""

        dst_dir  = "/opt/ptp_datasets/"
        dst      = dst_dir + os.path.basename(self.xz_file)
        raw_resp = self._yes or input(f"Move {self.xz_file} to {dst_dir}? [Y/n] ") or "Y"

        # Move dataset to '/opt/ptp_datasets/' and add entry on the dataset
        # catalog at '/opt/ptp_datasets/README.md'
        if (self._yes or (raw_resp.lower() == 'y')):
            # Move
            shutil.move(self.xz_file, dst)
            # Add to catalog
            docs = Docs(cfg_path=dst_dir)
            docs.add_dataset(dst) # assume file has already moved to dst

    def catch(self, signum, frame):
        self.en_capture = False
        logger.info("Terminating acquisition of dataset")
        self.end_json_file()
        self.save_to_bin()
        self.move()
        logging.info("Run:\n./analyze.py -vvvv -f %s" %(
            os.path.basename(self.xz_file)))
        exit()

    def run(self):
        """Save/process timestamp sets and complementary data acquired serially
        """
        signal.signal(signal.SIGINT, self.catch)
        signal.signal(signal.SIGTERM, self.catch)

        # Use the reader class to post-process each set of timestamp in
        # real-time and to print the associated PTP metrics for debugging
        reader = Reader()

        self.start_json_file()

        logger.info("Starting capture")
        delay_cal_mode = self.roe.delay_cal_mode
        last_seq_id    = None
        debug_buffer   = list()
        count          = 0
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

                # Complementary/asynchronous data
                for key in self.async_data:
                    if (self.async_data[key]):
                        run_data[key] = self.async_data[key].popleft()

                # Sanity check on PTP sequenceId
                if ((last_seq_id is not None) and
                    (run_data['seq_id'] != ((last_seq_id + 1) % 2**16))):
                    logging.error(
                        "PTP sequence id gap: {:d} to {:d}".format(
                            last_seq_id,
                            run_data['seq_id']
                        ))
                    # If we've just switched from delay calibration mode, accept
                    # the sequenceId gap. If not, stop acquisition now.
                    if (self.roe.delay_cal_mode != delay_cal_mode):
                        logging.warning("Delay calibration mode switched from "
                                        " {} to {} - Accepting gap...".format(
                                            delay_cal_mode,
                                            self.roe.delay_cal_mode))
                        self.roe.delay_cal_mode = delay_cal_mode
                    else:
                        break
                last_seq_id = run_data['seq_id']

                if (logger.root.level == logging.DEBUG):
                    debug_buffer.append(run_data)

                    if (run_data["idx"] % 20 == 19):
                        df = pd.DataFrame(debug_buffer)
                        print(tabulate(df, headers='keys', tablefmt='psql'))
                        debug_buffer.clear()

                # Append to output JSON file
                self.save_json(run_data)
                count += 1
            else:
                time.sleep(0.1)

        self.end_json_file()
        self.save_to_bin()
        self.move()


