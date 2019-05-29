#!/usr/bin/env python

"""Acquisition of timestamps via UART
"""
import serial, time, json, logging, signal
from pprint import pprint, pformat


class Serial():
    """Serial capture of timestamps from testbed

    Args:
        device    : Target device ('bbu_uart', 'rru_uart' or 'rru2_uart')
        n_samples : Target number of samples (0 for infinity)

    """

    def __init__(self, device, n_samples):
        self.device    = device
        self.n_samples = n_samples

        # Initializing serial connection
        self.connect(device)

        # Enable
        self.en_capture = True
        self.idx        = 0

        # Filename
        path = "data/"
        self.filename = path + "serial-" + time.strftime("%Y%m%d-%H%M%S") + ".json"

    def connect(self, device):
        """Try to establish a serial connection to a given device.

        Args:
            device : Target UART device within /dev
        """

        devices_list = ['bbu_uart',
                        'rru_uart',
                        'rru2_uart']

        assert(device in devices_list), "Unknown UART device"

        serial_args = '/dev/' + device

        try:
            self.serialdevice = serial.Serial(serial_args,
                                              baudrate= 115200, \
                                              bytesize= serial.EIGHTBITS, \
                                              parity= serial.PARITY_NONE, \
                                              stopbits= serial.STOPBITS_ONE,
                                              timeout = 1)
        except serial.serialutil.SerialException:
            logging.error('Error on %s connection' %(device))
        else:
            logging.info("Connected %s" %(device))

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
                fd.write(', ')
            json.dump(data, fd)

    def catch(self, signum, frame):
        self.end_json_array()
        logging.info("Terminating acquisition")
        exit()

    def run(self):
        """Continuously read the serial input and collect timestamps
        """
        signal.signal(signal.SIGINT, self.catch)
        signal.siginterrupt(signal.SIGINT, False)

        self.start_json_array()

        logging.info("Starting capture")
        while self.en_capture == True and \
              ((self.idx < self.n_samples) or self.n_samples == 0):

            line     = self.serialdevice.readline()
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

                logging.debug(pformat(run_data))
                logging.info("Index %d" %self.idx)

                # Append to output file
                self.save(run_data)
                self.idx += 1

        self.end_json_array()
