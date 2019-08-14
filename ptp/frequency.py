"""Estimators
"""
import logging
logger = logging.getLogger(__name__)


class Estimator():
    """Frequency offset estimator"""
    def __init__(self, data, delta=1):
        """Initialize the frequency offset estimator

        Args:
            data  : Array of objects with simulation data
            delta : (int > 0) Observation interval in samples. When set to 1,
                    estimates the frequency offsets based on consecutive data
                    entries.  When set to 2, estimates frequency offset i based
                    on timestamps from the i-th iteration and from iteration
                    'i-2', and so on.

        """
        assert(delta > 0)
        assert(isinstance(delta, int))
        self.data   = data
        self.delta  = delta

    def process(self):
        """Process the data

        Estimate the frequency offset relative to the reference over all the
        data. Do so by comparing interval measurements of the slave and the
        master. This is equivalent to differentiating the time offset
        measurements.

        """

        logger.info("Processing")

        for i,r in enumerate(self.data):
            idx = r["idx"]

            # Start estimating after accumulating enough history
            if (i < self.delta):
                continue

            # Estimate
            t1           = r["t1"]
            t2           = r["t2"]
            i_past       = i - self.delta
            t1_past      = self.data[i_past]["t1"]
            t2_past      = self.data[i_past]["t2"]
            delta_slave  = float(t2 - t2_past)
            delta_master = float(t1 - t1_past)
            y_est        = (delta_slave - delta_master) / delta_master

            logger.debug("Delta t2: %f ns\tDelta t1: %f ns\tFreq Offset: %f ppb" %(
                delta_slave, delta_master, y_est*1e9))

            r["y_est"] = y_est

    def set_truth(self, delta=4):
        """Set "true" frequency offset based on "true" time offset measurements

        Args:
            delta : (int > 0) Observation interval in samples. When set to 1,
                    estimates the frequency offsets based on consecutive data
                    entries.  When set to 2, estimates frequency offset i based
                    on timestamps from the i-th iteration and from iteration
                    'i-2', and so on (default: 4)
        """

        for i,r in enumerate(self.data):
            idx = r["idx"]

            # Start estimating after accumulating enough history
            if (i < delta):
                continue

            # Estimate
            t1           = r["t1"]
            x            = r["x"]
            i_past       = i - delta
            t1_past      = self.data[i_past]["t1"]
            x_past       = self.data[i_past]["x"]
            delta_x      = (x - x_past)
            delta_master = float(t1 - t1_past)
            y            = delta_x / delta_master

            logger.debug("True Freq. Offset: Delta x: %f ns\tDelta t1: %f ns\tFreq Offset: %f ppb" %(
                delta_x, delta_master, y*1e9))

            # Add to dataset
            r["rtc_y"] = y
