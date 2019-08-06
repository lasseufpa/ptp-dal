#!/usr/bin/env python

"""Acquisition of timestamps via UART
"""
import serial, time, json, logging, signal
from pprint import pprint, pformat


class Serial():
    def __init__(self, fpga_dev, sensor_dev, n_samples):
        """Serial capture of timestamps from testbed

        Args:
            fpga_dev   : FPGA device ('bbu_uart', 'rru_uart' or 'rru2_uart')
            sensor_dev : Sensor device ('roe_sensor')
            n_samples  : Target number of samples (0 for infinity)

        """
        self.n_samples = n_samples

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
        logging.info("Connected to %s" %(device))
        return serial_conn

    def start_json_array(self):
        """Start the output JSON array"""

        with open(self.filename, 'a') as fd:
            fd.write('[')

    def end_json_array(self):
        """Start the output JSON array"""

        with open(self.filename, 'a') as fd:
            fd.write(']')

    def save(self, data):
        """Save runner data on JSON file"""

        with open(self.filename, 'a') as fd:
            if (self.idx > 0):
                fd.write(',\n')
            json.dump(data, fd)

    def catch(self, signum, frame):
        self.end_json_array()
        logging.info("Terminating acquisition of %s" %(self.filename))
        exit()

    def run(self):
        """Continuously read the serial input and collect timestamps
        """
        signal.signal(signal.SIGINT, self.catch)
        signal.siginterrupt(signal.SIGINT, False)

        self.start_json_array()

        logging.info("Starting capture")
        temperature = None
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
            line     = self.fpga.readline()
            line_key = line.decode().split(" ")[0]
            line_val = line.decode().split(" ")

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
                t1_pps     = int(line_val[18],16)
                t1_pps_sec = int(line_val[20],16)
                t4_pps     = int(line_val[22],16)
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
                    't1_pps'     : t1_pps,
                    't1_pps_sec' : t1_pps_sec,
                    't4_pps'     : t4_pps,
                    't4_pps_sec' : t4_pps_sec
                }

                # Append the temperature
                if (temperature is not None):
                    run_data["temp"] = temperature

                logging.debug(pformat(run_data))
                logging.info("Index %d" %self.idx)

                # Append to output file
                self.save(run_data)
                self.idx += 1

        self.end_json_array()
