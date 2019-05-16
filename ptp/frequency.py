"""Estimators
"""
import logging


class Estimator():
    """Frequency offset estimator"""
    def __init__(self, data, period_ns=1e9):
        """Initialize the frequency offset estimator

        Args:
            data      : Array of objects with simulation data
            period_ns : Periodicity of freq. offset estimations in ns
        """
        self.data      = data
        self.period_ns = period_ns

    def process(self):
        """Process the data

        Estimate the frequency offset relative to the reference over all the
        data.

        """

        last_t1 = None
        last_t2 = None

        for r in self.data:
            t1 = r["t1"]
            t2 = r["t2"]

            # Start by saving the first value
            if (last_t1 is None or last_t2 is None):
                last_t1 = t1
                last_t2 = t2
                continue

            # Is it time to compute a frequency offset already?
            if (float(t1 - last_t1) < self.period_ns):
                continue

            # Estimate
            delta_slave   = float(t2 - last_t2)
            delta_master  = float(t1 - last_t1)
            y_est         = (delta_slave - delta_master) / delta_master

            logger = logging.getLogger("FreqEstimator")
            logger.debug("Delta t2: %f ns\tDelta t1: %f ns\tFreq Offset: %f ppb" %(
                delta_slave, delta_master, y_est*1e9))

            # Save for the next estimation
            last_t1 = t1
            last_t2 = t2

            r["y_est"] = y_est

