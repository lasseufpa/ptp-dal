"""Estimators
"""
import logging


class FreqEstimator():
    """Frequency offset estimator"""
    def __init__(self, period_ns):
        """Initialize the frequency offset estimator

        Args:
            period_ns : Periodicity of estimations in ns
        """
        self.period_ns  = period_ns
        self.last_t1 = None
        self.last_t2 = None

    def process(self, t1, t2):
        """Estimate the frequency offset relative to the reference

        Returns:
            The frequency offset as a float
        """

        # Start by saving the first value
        if (self.last_t1 is None or self.last_t2 is None):
            self.last_t1 = t1
            self.last_t2 = t2
            return

        # Is it time to compute a frequency offset already?
        if (float(t1 - self.last_t1) < self.period_ns):
            return

        # Estimate
        delta_slave   = float(t2 - self.last_t2)
        delta_master  = float(t1 - self.last_t1)
        y_est         = (delta_slave - delta_master) / delta_master

        logger = logging.getLogger("FreqEstimator")
        logger.debug("Delta t2: %f ns\tDelta t1: %f ns\tFreq Offset: %f ppb" %(
            delta_slave, delta_master, y_est*1e9))

        # Save for the next iteration
        self.last_t1 = t1
        self.last_t2 = t2

        return y_est
