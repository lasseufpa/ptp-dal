#!/usr/bin/env python

"""Acquisition of timestamps via UART
"""
import serial, time, json, logging, signal, os
from pprint import pprint, pformat
from ptp.reader import Reader
from ptp.docs import Docs


logger = logging.getLogger(__name__)


class Serial():
    def __init__(self, fpga_dev, sensor_dev, n_samples, metadata):
        """Serial capture of timestamps from testbed

        Args:
            fpga_dev   : FPGA device ('bbu_uart', 'rru_uart' or 'rru2_uart')
            sensor_dev : Sensor device ('roe_sensor')
            n_samples  : Target number of samples (0 for infinity)
            metadata   : Information about the testbed configuration

        """
        self.n_samples = n_samples
        self.metadata  = metadata

        # Initializing serial connection
        self.fpga = self.connect(fpga_dev)

        if (sensor_dev is not None):
            self.sensor = self.connect(sensor_dev)
        else:
            self.sensor = None

        # Enable
        self.en_capture = True
        self.idx        = 0

        # Filename
        path = "data/"
        self.filename = path + "serial-" + time.strftime("%Y%m%d-%H%M%S") + ".json"

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
            if (self.idx > 0):
                fd.write(',\n')
            json.dump(data, fd)

    def move(self, file):
        """Move JSON file"""

        dst = "/opt/ptp_datasets/" + os.path.basename(file)
        raw_resp = input(f"Move {file} to {dst}? [Y/n] ") or "Y"
        response = raw_resp.lower()

        if (response == 'y'):
            os.rename(file, dst)

            # Add metadata to '/opt/ptp_datasets/README.md'
            Docs.add_value(self.filename)

    def catch(self, signum, frame):
        self.end_json_file()
        logger.info("Terminating acquisition of %s" %(self.filename))
        self.move()
        exit()

    def run(self, print_en, capture_occ=True):
        """Continuously read the serial input and collect timestamps

        Args:
            print_en    : Whether to print non-timestamp logs to stdout
            capture_occ : Whether to capture the RoE DAC interface occupancy

        """
        signal.signal(signal.SIGINT, self.catch)
        signal.siginterrupt(signal.SIGINT, False)

        format_str = ('i:{:>4d} t1:{:>5d},{:>9d} t2:{:>5d},{:>9d} '
                      't3:{:>5d},{:>9d} t4:{:>5d},{:>9d} '
                      't1_pps:{:>5d},{:>9d} t4_pps:{:>5d},{:>9d} '
                      'temp:{:>4.1f} occ:{:>4d}')

        # Use the reader class to post-process each set of timestamp in
        # real-time and to print the associated PTP metrics
        reader = Reader()

        self.start_json_file()

        logger.info("Starting capture")
        temperature = None
        occupancy   = None
        while self.en_capture == True and \
              ((self.idx < self.n_samples) or self.n_samples == 0):

            # Read the temperature every time there is some spare time
            # from reading the FPGA (when its serial input buffer is empty)
            if (self.sensor is not None and (self.fpga.in_waiting == 0)):
                temperature_str = self.sensor.readline()
                if (len(temperature_str) > 0):
                    temperature = float(temperature_str)
                    # Reset input buffer so that measurements don't accumulate
                    # and we read the up-to-date temperature.
                self.sensor.reset_input_buffer()

            # Read timestamps from FPGA
            line     = self.fpga.readline().decode()
            line_key = line.split(" ")[0]
            line_val = line.split(" ")

            if capture_occ and "Occupancy" in line:
                split_line = line.split("\t")
                if (len(split_line) > 1):
                    try:
                        occupancy = int(split_line[1])
                    except ValueError:
                        occupancy = None

            if line_key == "Timestamps":
                # Normal PTP Timestamps
                t1_ns  = int(line_val[2],16)
                t2_ns  = int(line_val[4],16)
                t3_ns  = int(line_val[6],16)
                t4_ns  = int(line_val[8],16)
                t1_sec = int(line_val[10],16)
                t2_sec = int(line_val[12],16)
                t3_sec = int(line_val[14],16)
                t4_sec = int(line_val[16],16)

                # PPS Timestamps
                t1_pps_ns  = int(line_val[18],16)
                t1_pps_sec = int(line_val[20],16)
                t4_pps_ns  = int(line_val[22],16)
                t4_pps_sec = int(line_val[24],16)

                # Append to results
                run_data = {
                    "idx"        : self.idx,
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

                # Process PTP metrics for debugging
                reader.process(run_data)

                # Append the temperature
                if (temperature is not None):
                    run_data["temp"] = temperature

                # Append the occupancy
                if (occupancy is not None):
                    run_data["occupancy"] = occupancy

                logger.debug(format_str.format(self.idx, t1_sec, t1_ns, t2_sec,
                                               t2_ns, t3_sec, t3_ns, t4_sec,
                                               t4_ns, t1_pps_sec, t1_pps_ns,
                                               t4_pps_sec, t4_pps_ns,
                                               temperature or -1,
                                               occupancy or -1))

                # Append to output file
                self.save(run_data)
                self.idx += 1
            elif (print_en):
                print(line, end='')

        self.end_json_file()
        self.move()
