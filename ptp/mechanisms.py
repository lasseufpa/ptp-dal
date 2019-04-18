"""PTP packet exchange mechanisms
"""
import logging


class DelayReqResp():
    def __init__(self, seq_num, t1):
        """Delay request-response mechanism

        Args:
            seq_num : Sequence number
            t1      : Sync departure timestamp
        """
        self.seq_num = seq_num
        self.t1      = t1
        self.t2      = None
        self.t3      = None
        self.t4      = None

    def set_t2(self, seq_num, t2):
        """Set Sync arrival timestamp

        Args:
            seq_num : Sequence number
            t2      : Sync arrival timestamp
        """
        assert(self.seq_num == seq_num)
        self.t2      = t2

    def set_t3(self, seq_num, t3):
        """Set Delay_Req departure timestamp

        Args:
            seq_num : Sequence number
            t3      : Delay_Req departure timestamp
        """
        assert(self.seq_num == seq_num)
        self.t3      = t3

    def set_t4(self, seq_num, t4):
        """Set Delay_Req departure timestamp

        Args:
            seq_num : Sequence number
            t4      : Delay_Req departure timestamp
        """
        assert(self.seq_num == seq_num)
        self.t4      = t4

    def _estimate_delay(self, t1_ns, t2_ns, t3_ns, t4_ns):
        """Estimate the one-way delay

        Args:
            t1_ns : Sync departure timestamp from Master
            t2_ns : Sync arrival timestamp from Slave
            t3_ns : Delay_Req departure timestamp from Slave
            t4_ns : Delay_Req arrival timestamp from Master

        Returns:
            The delay estimation in ns
        """

        t4_minus_t1 = t4_ns - t1_ns

        # If the ns counter wraps, this difference wold become negative.
        # In this case, add one second back:
        if (t4_minus_t1 < 0):
            t4_minus_t1 = t4_minus_t1 + 1e9

        t3_minus_t2 = t3_ns - t2_ns
        # If the ns counter wraps, this difference wold become negative.
        # In this case, add one second back:
        if (t3_minus_t2 < 0):
            t3_minus_t2 = t3_minus_t2 + 1e9

        delay_est_ns = (t4_minus_t1 - t3_minus_t2) / 2
        return delay_est_ns

    def process(self):
        """Process all four timestamps"""
        delay  = self._estimate_delay(self.t1.ns, self.t2.ns, self.t3.ns,
                                      self.t4.ns)
        logger = logging.getLogger("DelayReqResp")
        logger.info("seq_num #%d\tt1: %s\tt2: %s\tt3: %s\tt4: %s\tdelay: %u ns" %(
            self.seq_num, self.t1, self.t2, self.t3, self.t4, delay))
