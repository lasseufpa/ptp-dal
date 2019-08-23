"""Estimators
"""
import numpy as np
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

        logger.info("Processing with N=%d" %(self.delta))

        # Remove previous estimates in case they already exist
        for r in self.data:
            r.pop("y_est", None)

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

    def optimize(self):
        """Optimize observation interval for minimum MSE

        """

        log_min_window = 1
        log_max_window = 16
        log_window_len = np.arange(log_min_window, log_max_window + 1, 1)
        window_len     = 2**log_window_len

        logger.info("Optimize observation window" %())
        logger.info("Try from N = %d to N = %d" %(
            2**log_min_window, 2**log_max_window))

        min_y_mse  = 1e9
        N_opt      = 0

        for N in window_len:
            for r in self.data:
                r.pop("y_est", None)

            self.delta = N
            self.process()

            y_err = np.array([1e9*(r["y_est"] - r["rtc_y"]) for r in self.data
                              if ("y_est" in r and "rtc_y" in r)])
            y_mse = np.square(y_err).mean()

            if (y_mse < min_y_mse):
                N_opt     = N
                min_y_mse = y_mse

        logger.info("Minimum MSE: %f ppb" %(min_y_mse))
        logger.info("Optimum N:   %d" %(N_opt))
        self.delta = N_opt

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

    def estimate_drift(self):
        """Estimate the incremental drifts due to frequency offset

        On each iteration, the true time offset changes due to the instantaneous
        frequency offset. On iteration n, with freq. offset y[n], it will
        roughly change w.r.t. the previous iteration by:

        drift[n] = y[n] * (t1[n] - t1[n-1])

        Estimate these incremental changes and save on the dataset.

        """

        # Compute the drift within the observation window
        for i,r in enumerate(self.data[1:]):
            if ("y_est" in r):
                delta = float(r["t1"] - self.data[i]["t1"])
                # NOTE: index i is the enumerated index, not the data entry
                # index. Since we get self.data[1:] (i.e. starting from index
                # 1), "i" lags the actual data index by 1.
                r["drift"] = r["y_est"] * delta


