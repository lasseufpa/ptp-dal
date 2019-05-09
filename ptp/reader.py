import logging, json
import numpy as np
from ptp.timestamping import Timestamp
from ptp.mechanisms import *


class Reader():
    """Reader of data acquired through the test

    Captures data in real-time via serial communication or reads a log file with
    previously acquired data. Saves the data into a list of dictionaries that
    contain the same keys as produced by the PTP runner.

    """
    def __init__(self, log_file=None):
        """Constructor

        Args:
            log_file : JSON log file to read from

        """
        self.running  = True
        self.data     = list()
        self.log_file = log_file

    def process(self, max_len, infer_secs):
        """Load the JSON data into self.data

        Args:
            max_len    : Maximum number of entries to process
            infer_secs : Ignore acquired seconds and infer their values instead

        """

        with open(self.log_file) as fin:
            data = json.load(fin)

        DelayReqResp(0,0).log_header()

        if (max_len > 0):
            n_data = max_len
        else:
            n_data = len(data)

        if (infer_secs):
            last_t1_ns = 0
            last_t2_ns = 0
            last_t3_ns = 0
            last_t4_ns = 0
            t1_sec     = 0
            t2_sec     = 0
            t3_sec     = 0
            t4_sec     = 0

        # Put info in dictionary and append to self.data
        for i in range (0, n_data):
            idx = data[i]["idx"]

            if (infer_secs):
                # Ns values
                t1_ns = data[i]["t1"]
                t2_ns = data[i]["t2"]
                t3_ns = data[i]["t3"]
                t4_ns = data[i]["t4"]

                # Has ns wrapped around?
                if (t1_ns < last_t1_ns):
                    t1_sec += 1
                if (t2_ns < last_t2_ns):
                    t2_sec += 1
                if (t3_ns < last_t3_ns):
                    t3_sec += 1
                if (t4_ns < last_t4_ns):
                    t4_sec += 1

                # Corresponding timestamp instances
                t1  = Timestamp(t1_sec, t1_ns)
                t2  = Timestamp(t2_sec, t2_ns)
                t3  = Timestamp(t3_sec, t3_ns)
                t4  = Timestamp(t4_sec, t4_ns)

                # Save ns values for next iteration (for wrapping detection)
                last_t1_ns = t1_ns
                last_t2_ns = t2_ns
                last_t3_ns = t3_ns
                last_t4_ns = t4_ns
            else:
                t1  = Timestamp(data[i]["t1_sec"], data[i]["t1"])
                t2  = Timestamp(data[i]["t2_sec"], data[i]["t2"])
                t3  = Timestamp(data[i]["t3_sec"], data[i]["t3"])
                t4  = Timestamp(data[i]["t4_sec"], data[i]["t4"])

            # Create a delay request-response instance
            dreqresp = DelayReqResp(idx, t1)

            # Define its associated emtrics
            dreqresp.set_t2(idx, t2)
            dreqresp.set_t3(idx, t3)
            dreqresp.set_t4(idx, t4)

            # Process and put results within self.data
            results = dreqresp.process()
            self.data.append(results)

        return data
