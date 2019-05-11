import logging, json
import numpy as np
from ptp.timestamping import Timestamp
from ptp.mechanisms import *
from ptp.estimators import *


class Reader():
    """Reader of data acquired through the test

    Captures timestamps in real-time via serial communication or reads a log
    file with previously acquired timestamps. Post-processes the timestamps
    following the same sequence that is adopted by the PTP runner and similarly
    saves reults into a list of dictionaries containing the same keys as
    produced by the runner.

    """
    def __init__(self, log_file=None, freq_est_per = 1e9):
        """Constructor

        Args:
            log_file     : JSON log file to read from
            freq_est_per : Raw freq. estimation period in ns

        """
        self.running         = True
        self.data            = list()
        self.log_file        = log_file
        self.freq_est_per_ns = freq_est_per

    def process(self, max_len, infer_secs):
        """Loads timestamps and post-processes to generate PTP data

        First loads a list containing sets of timestamps (t1, t2, t3 and t4)
        from a JSON. Then, iteratively applies the timestamps into delay
        request-response mechanism objects, using the latter to post-process the
        timestamps and obtain PTP metrics. Save each metric into self.data just
        like the PTP runner.

        Args:
            max_len    : Maximum number of entries to process
            infer_secs : Ignore acquired seconds and infer their values instead

        """

        with open(self.log_file) as fin:
            data = json.load(fin)

        # Raw frequency offset estimator
        freq_estimator = FreqEstimator(self.freq_est_per_ns)

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

            # Estimate frequency offset
            y_est = freq_estimator.process(dreqresp.t1, dreqresp.t2)
            if (y_est is not None):
                results["y_est"] = y_est

            self.data.append(results)

        return data
