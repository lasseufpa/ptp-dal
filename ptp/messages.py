"""PTP Messages
"""
import logging, heapq
import numpy as np


class PtpEvt():
    def __init__(self, name, period_sec=None, pdv_distr="Gamma"):
        """PTP Event Message

        Controls transmission and reception of a PTP event message. When the
        message is periodically transmitted (Sync), a period must be passed by
        argument. Otherwise, transmission must be scheduled manually.

        Args:
            name       : Message name
            period_sec : Transmission period in seconds
            pdv_distr  : PDV distribution

        """

        self.name          = name
        self.period_sec    = period_sec
        self.on_way        = False
        self.next_tx       = float("inf")
        self.next_rx       = float("inf")
        self.seq_num       = None
        self.tx_tstamp     = None
        self.rx_tstamp     = None
        self.one_way_delay = None
        self.pdv_distr     = pdv_distr

        assert(pdv_distr == "Gamma" or pdv_distr == "Gaussian")

    def _sched_next_tx(self, tx_sim_time):
        """Compute next transmission time for periodic message

        Args:
            tx_sim_time : Simulation time (secs) corresponding to the Tx instant

        """

        uncertainty_ns = np.random.normal(0, 1500)
        # NOTE: we've measured around 1.5 microsecs of uncertainty on the
        # interval between consecutive t1s

        self.next_tx = tx_sim_time + self.period_sec + (uncertainty_ns * 1e-9)

    def _sched_rx(self, tx_sim_time):
        """Schedule Reception

        Args:
            tx_sim_time : Simulation time (secs) corresponding to the Tx instant

        """

        if (self.pdv_distr == "Gamma"):
            delay_ns = np.random.gamma(shape=2, scale=1000)
            # FIXME set Gamma params
        elif (self.pdv_distr == "Gaussian"):
            delay_ns = np.random.normal(loc=2000, scale=200)
            # FIXME set Gaussian params

        self.next_rx = tx_sim_time + (delay_ns * 1e-9)

        logger = logging.getLogger("PtpEvt")
        logger.debug("Delay of %s #%d: %f ns" %(self.name, self.seq_num, delay_ns))

    def sched_tx(self, tx_sim_time, evts):
        """Manually schedule a transmission time

        Args:
            tx_sim_time : Target simulation time (secs) for Tx
            evts        : Event heap queue

        """
        assert (self.period_sec is None), \
            "Only non-periodic PTP msgs should be scheduled"

        self.next_tx = tx_sim_time
        heapq.heappush(evts, self.next_tx)

        logger = logging.getLogger("PtpEvt")
        logger.debug("Schedule %s transmission to %f ns" %(self.name,
                                                           self.next_tx * 1e9))

    def tx(self, sim_time, rtc_timestamp, evts):
        """Transmit message

        Args:
            sim_time      : Simulation time in seconds
            rtc_timestamp : RTC Time
            evts          : Event heap queue

        Returns:
            True when effectively transmitted
        """

        # Do not transmit before scheduled time or if there is already an
        # ongoing transmission
        if ((self.next_tx is None) or (sim_time < self.next_tx) or self.on_way):
            return False

        # Proceed with transmission
        self.on_way         = True
        self.tx_tstamp      = rtc_timestamp

        if (self.seq_num is None):
            self.seq_num = 0
        else:
            self.seq_num += 1

        logger = logging.getLogger("PtpEvt")
        logger.debug("Transmitting %s #%d at %s" %(self.name, self.seq_num,
                                                   sim_time))

        # Schedule the next transmission for periodic messages. For non-periodic
        # messages, just clear the next Tx time.
        if (self.period_sec is not None):
            self._sched_next_tx(sim_time)
            heapq.heappush(evts, self.next_tx)
        else:
            self.next_tx = None

        # Schedule the reception
        self._sched_rx(sim_time)
        heapq.heappush(evts, self.next_rx)

        return True

    def rx(self, sim_time, rx_rtc_tstamp, tx_rtc_tstamp):
        """Receive Message

        Process the reception of the message. Take a timestamp from the RTC of
        the receiver, and also measure the true one-way delay of the message by
        using a snapshot from the RTC of the message transmitter.

        Args:
            sim_time      : Simulation time in seconds
            rx_rtc_tstamp : Timestamp from RTC of message receiver
            tx_rtc_tstamp : Timestamp from RTC of message transmitter

        Returns:
            True when effectively received

        """

        # Do not receive before scheduled time or if there isn't a message on
        # the way
        if ((sim_time < self.next_rx) or (not self.on_way)):
            return False

        # Proceed with reception
        self.on_way         = False
        self.rx_tstamp      = rx_rtc_tstamp
        self.one_way_delay  = float(tx_rtc_tstamp - self.tx_tstamp)

        logger = logging.getLogger("PtpEvt")
        logger.debug("Received %s #%d at %s" %(self.name, self.seq_num,
                                               sim_time))
        logger.debug("One-way delay: %f" %(self.one_way_delay))

        return True
