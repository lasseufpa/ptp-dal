"""PTP packet exchange mechanisms
"""
import logging

logger = logging.getLogger("DelayReqResp")


class DelayReqResp():
    """Delay request-response mechanism

    Args:
        seq_num : Sequence number
        t1      : Sync departure timestamp
    """
    def __init__(self, seq_num, t1):

        self.seq_num = seq_num
        self.t1 = t1
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.d_fw = None
        self.d_bw = None
        self.toffset = None
        self.asymmetry = None

    @staticmethod
    def log_header(level=logging.DEBUG, logger=logger):
        """Print logging header

        Args:
            level  : Logging level to use when printing
            logger : Logger object

        """

        if (level == logging.INFO):
            print_fn = logger.info
        else:
            print_fn = logger.debug

        print_fn(("-----------------------------------------------"
                  "---------------------------------"))
        header = '{:>4} {:^12} {:^12} {:^9} {:^9} {:^9} {:^9} {:^9}'.format(
            "idx", "x_est", "x", "x_est_err", "delay_est", "d_m2s", "d_s2m",
            "asym")
        print_fn(header)
        print_fn(("-----------------------------------------------"
                  "---------------------------------"))

    def set_t2(self, seq_num, t2):
        """Set Sync arrival timestamp

        Args:
            seq_num : Sequence number
            t2      : Sync arrival timestamp
        """
        assert (self.seq_num == seq_num)
        self.t2 = t2

    def set_t3(self, seq_num, t3):
        """Set Delay_Req departure timestamp

        Args:
            seq_num : Sequence number
            t3      : Delay_Req departure timestamp
        """
        assert (self.seq_num == seq_num)
        self.t3 = t3

    def set_t4(self, seq_num, t4):
        """Set Delay_Req departure timestamp

        Args:
            seq_num : Sequence number
            t4      : Delay_Req departure timestamp
        """
        assert (self.seq_num == seq_num)
        self.t4 = t4

    def set_forward_delay(self, seq_num, delay):
        """Save the "true" master-to-slave one-way delay

        This truth comes from the difference of RTC timestamps. Hence, although
        close, it still suffers from uncertainties and quantization.

        Args:
            seq_num : Sequence number
            delay   : Master-to-slave delay

        """
        assert (self.seq_num == seq_num)
        self.d_fw = delay

        # Update the true delay asymmetry:
        if (self.d_bw is not None):
            self.asymmetry = (self.d_fw - self.d_bw) / 2

    def set_backward_delay(self, seq_num, delay):
        """Save the "true" slave-to-master one-way delay

        This truth comes from the difference of RTC timestamps.

        Args:
            seq_num : Sequence number
            delay   : Slave-to-master delay

        """
        assert (self.seq_num == seq_num)
        self.d_bw = delay

        # Update the true delay asymmetry:
        if (self.d_fw is not None):
            self.asymmetry = (self.d_fw - self.d_bw) / 2

    def _estimate_delay(self):
        """Estimate the one-way delay

        Returns:
            The delay estimation in ns as a float
        """
        delay_est_ns = (float(self.t4 - self.t1) -
                        float(self.t3 - self.t2)) / 2
        return delay_est_ns

    def _estimate_time_offset(self):
        """Estimate the time offset from master

        Returns:
            The time offset as a Timestamp object
        """
        offset_from_master = ((self.t2 - self.t1) - (self.t4 - self.t3)) / 2
        return offset_from_master

    def set_true_toffset(self, master_tstamp, slave_tstamp):
        """Save the true time offset

        Given two simultaneously-taken timestamps from master and slave clocks,
        compute the true time offset at the instant corresponding to the given
        timestamps.

        Args:
            master_tstamp : Timestamp from the master RTC
            slave_tstamp  : Timestamp from the slave RTC

        """

        self.toffset = slave_tstamp - master_tstamp

    def process(self):
        """Process all four timestamps

        Wrap-up the delay request-response exchange by computing the associated
        metrics with the four collected timestamps.

        Returns:
            Dictionary with resulting metrics

        """

        # Estimations
        delay_est = self._estimate_delay()
        toffset_est = float(self._estimate_time_offset())

        # Save all relevant metrics on a dictionary
        results = {
            "idx": self.seq_num,
            "t1": self.t1,
            "t2": self.t2,
            "t3": self.t3,
            "t4": self.t4,
            "d": self.d_fw,  # Sync one-way delay
            "d_bw": self.d_bw,  # Delay_Req one-way delay
            "d_est": delay_est,
            "x_est": toffset_est,
            "x": None,
            "x_est_err": None,
            "asym": self.asymmetry
        }

        if (self.toffset is not None):
            # Time offset estimation error
            toffset_err = toffset_est - float(self.toffset)
            # Save on results
            results["x"] = float(self.toffset)
            results["x_est_err"] = toffset_err

        return results

    @staticmethod
    def log(r, level=logging.DEBUG, logger=logger):
        """Print results

        Args:
            r      : Dictionary with results
            level  : Logging level to use when printing
            logger : Logger object

        """

        if (level == logging.INFO):
            print_fn = logger.info
        else:
            print_fn = logger.debug

        print_fn(('{:^4d} {:^ 12.1f} {:^ 12.1f} '
                  '{:^ 9.1f} {:^9.1f} '
                  '{:^9.1f} {:^9.1f} '
                  '{:^ 9.1f}').format(r['idx'], r['x_est'], float(r['x'] or 0),
                                      float(r['x_est_err'] or 0), r['d_est'],
                                      float(r['d'] or 0), float(r['d_bw']
                                                                or 0),
                                      float(r['asym'] or 0)))
