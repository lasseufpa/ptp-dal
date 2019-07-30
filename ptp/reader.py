import logging, json
import numpy as np
from ptp.timestamping import Timestamp
from ptp.mechanisms import *


class Reader():
    """Reader of data acquired through the test

    Captures timestamps in real-time via serial communication or reads a log
    file with previously acquired timestamps. Post-processes the timestamps
    following the same sequence that is adopted by the PTP runner and similarly
    saves reults into a list of dictionaries containing the same keys as
    produced by the runner.

    """
    def __init__(self, log_file=None):
        """Constructor

        Args:
            log_file     : JSON log file to read from

        """
        self.running         = True
        self.data            = list()
        self.log_file        = log_file

    def process(self, max_len=0, infer_secs=True, no_pps=False, reverse_ms=True):
        """Loads timestamps and post-processes to generate PTP data

        First loads a list containing sets of timestamps (t1, t2, t3 and t4)
        from a JSON. Then, iteratively applies the timestamps into delay
        request-response mechanism objects, using the latter to post-process the
        timestamps and obtain PTP metrics. Save each metric into self.data just
        like the PTP runner.

        Args:
            max_len    : Maximum number of entries to process
            infer_secs : Ignore acquired seconds and infer their values instead
            no_pps     : Logs to be processed do not contain the reference
                         timestamps acquired from the PPS RTC.
            reverse_ms : Reverse the master-to-slave direction for offset
                         computations. This is used when the PDelay
                         request-response that is being processed is originated
                         at the slave, so that t1 and t4 are slave timestamps.

        """

        with open(self.log_file) as fin:
            data = json.load(fin)

        # Debug print header
        DelayReqResp(0,0).log_header()

        # Restrict number of iterations, if so desired
        if (max_len > 0):
            n_data = max_len
        else:
            n_data = len(data)

        # Prepare to infer seconds, if so desired
        if (infer_secs):
            last_t1_ns = 0
            last_t2_ns = 0
            last_t3_ns = 0
            last_t4_ns = 0
            t1_sec     = 0
            t2_sec     = 0
            t3_sec     = 0
            t4_sec     = 0
            last_t1_pps_ns = 0
            last_t4_pps_ns = 0
            t1_pps_sec     = 0
            t4_pps_sec     = 0

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

                # PPS timestamps
                if (not no_pps):
                    t1_pps_ns = data[i]["t1_pps"]
                    t4_pps_ns = data[i]["t4_pps"]

                    if (t1_pps_ns < last_t1_pps_ns):
                        t1_pps_sec += 1
                    if (t4_pps_ns < last_t4_pps_ns):
                        t4_pps_sec += 1

                    t1_pps = Timestamp(t1_pps_sec, t1_pps_ns)
                    t4_pps = Timestamp(t4_pps_sec, t4_pps_ns)

                    last_t1_pps_ns = t1_pps_ns
                    last_t4_pps_ns = t4_pps_ns
            else:
                t1  = Timestamp(data[i]["t1_sec"], data[i]["t1"])
                t2  = Timestamp(data[i]["t2_sec"], data[i]["t2"])
                t3  = Timestamp(data[i]["t3_sec"], data[i]["t3"])
                t4  = Timestamp(data[i]["t4_sec"], data[i]["t4"])

                if (not no_pps):
                    t1_pps  = Timestamp(data[i]["t1_pps_sec"],
                                        data[i]["t1_pps"])
                    t4_pps  = Timestamp(data[i]["t4_pps_sec"],
                                        data[i]["t4_pps"])

            # Add timestamps to delay req-resp
            if (reverse_ms):
                dreqresp = DelayReqResp(idx, t3)
                dreqresp.set_t2(idx, t4)
                dreqresp.set_t3(idx, t1)
                dreqresp.set_t4(idx, t2)
            else:
                dreqresp = DelayReqResp(idx, t1)
                dreqresp.set_t2(idx, t2)
                dreqresp.set_t3(idx, t3)
                dreqresp.set_t4(idx, t4)

            # Set ground truth based on PPS timestamps
            if (not no_pps):
                if (reverse_ms):
                    forward_delay  = float(t4_pps - t3)
                    backward_delay = float(t2 - t1_pps)
                else:
                    forward_delay  = float(t2 - t1_pps)
                    backward_delay = float(t4_pps - t3)
                dreqresp.set_forward_delay(idx, forward_delay)
                dreqresp.set_backward_delay(idx, backward_delay)
                dreqresp.set_true_toffset(t4_pps, t4)

            # Process and put results within self.data
            results = dreqresp.process()
            self.data.append(results)

