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
    def __init__(self, log_file=None, infer_secs=True, no_pps=False,
                 reverse_ms=True):
        """Constructor

        Args:
            log_file   : JSON log file to read from
            infer_secs : Ignore acquired seconds and infer their values instead
            no_pps     : Logs to be processed do not contain the reference
                         timestamps acquired from the PPS RTC.
            reverse_ms : Reverse the master-to-slave direction for offset
                         computations. This is used when the PDelay
                         request-response that is being processed is originated
                         at the slave, so that t1 and t4 are slave timestamps.

        """
        self.running         = True
        self.data            = list()
        self.log_file        = log_file
        self.infer_secs      = infer_secs
        self.no_pps          = no_pps
        self.reverse_ms      = reverse_ms

        # Prepare to infer seconds, if so desired
        self.last_t1_ns     = 0
        self.last_t2_ns     = 0
        self.last_t3_ns     = 0
        self.last_t4_ns     = 0
        self.t1_sec         = 0
        self.t2_sec         = 0
        self.t3_sec         = 0
        self.t4_sec         = 0
        self.last_t1_pps_ns = 0
        self.last_t4_pps_ns = 0
        self.t1_pps_sec     = 0
        self.t4_pps_sec     = 0

        self.idx = 0

    def process(self, data):
        """Process a set of timestamps

        Apply the timestamps into a delay request-response mechanism object and
        use the latter to obtain PTP metrics.

        Args:
            data : dicitionary containing timestamp data

        Returns:
            results dictionary containing PTP sync metrics

        """

        self.idx = idx = data["idx"]

        # Print header periodically
        if (self.idx % 20 == 0):
            DelayReqResp.log_header()

        if (self.infer_secs):
            # Ns values
            t1_ns = data["t1"]
            t2_ns = data["t2"]
            t3_ns = data["t3"]
            t4_ns = data["t4"]

            # Has ns wrapped around?
            if (t1_ns < self.last_t1_ns):
                self.t1_sec += 1
            if (t2_ns < self.last_t2_ns):
                self.t2_sec += 1
            if (t3_ns < self.last_t3_ns):
                self.t3_sec += 1
            if (t4_ns < self.last_t4_ns):
                self.t4_sec += 1

            # Corresponding timestamp instances
            t1  = Timestamp(self.t1_sec, t1_ns)
            t2  = Timestamp(self.t2_sec, t2_ns)
            t3  = Timestamp(self.t3_sec, t3_ns)
            t4  = Timestamp(self.t4_sec, t4_ns)

            # Save ns values for next iteration (for wrapping detection)
            self.last_t1_ns = t1_ns
            self.last_t2_ns = t2_ns
            self.last_t3_ns = t3_ns
            self.last_t4_ns = t4_ns

            # PPS timestamps
            if (not self.no_pps):
                t1_pps_ns = data["t1_pps"]
                t4_pps_ns = data["t4_pps"]

                if (t1_pps_ns < self.last_t1_pps_ns):
                    self.t1_pps_sec += 1
                if (t4_pps_ns < self.last_t4_pps_ns):
                    self.t4_pps_sec += 1

                t1_pps = Timestamp(self.t1_pps_sec, t1_pps_ns)
                t4_pps = Timestamp(self.t4_pps_sec, t4_pps_ns)

                self.last_t1_pps_ns = t1_pps_ns
                self.last_t4_pps_ns = t4_pps_ns
        else:
            t1  = Timestamp(data["t1_sec"], data["t1"])
            t2  = Timestamp(data["t2_sec"], data["t2"])
            t3  = Timestamp(data["t3_sec"], data["t3"])
            t4  = Timestamp(data["t4_sec"], data["t4"])

            if (not self.no_pps):
                t1_pps  = Timestamp(data["t1_pps_sec"],
                                    data["t1_pps"])
                t4_pps  = Timestamp(data["t4_pps_sec"],
                                    data["t4_pps"])

        # Add timestamps to delay req-resp
        if (self.reverse_ms):
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
        if (not self.no_pps):
            if (self.reverse_ms):
                forward_delay  = float(t4_pps - t3)
                backward_delay = float(t2 - t1_pps)
            else:
                forward_delay  = float(t2 - t1_pps)
                backward_delay = float(t4_pps - t3)
            dreqresp.set_forward_delay(idx, forward_delay)
            dreqresp.set_backward_delay(idx, backward_delay)
            dreqresp.set_true_toffset(t4_pps, t4)

        # Process
        results = dreqresp.process()
        dreqresp.log(results)
        return results

    def run(self, max_len=0):
        """Loads timestamps and post-processes to generate PTP data

        Load a list containing sets of timestamps (t1, t2, t3 and t4) from a
        JSON and save each set of PTP metrics into self.data just like the PTP
        runner would.

        Args:
            max_len    : Maximum number of entries to process

        """

        with open(self.log_file) as fin:
            data = json.load(fin)

        # Debug print header
        DelayReqResp.log_header()

        # Restrict number of iterations, if so desired
        if (max_len > 0):
            n_data = max_len
        else:
            n_data = len(data)

        # Put info in dictionary and append to self.data
        for i in range (0, n_data):
            results = self.process(data[i])
            self.data.append(results)


