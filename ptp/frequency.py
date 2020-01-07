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

        t1           = np.array([float(r["t1"]) for r in self.data])
        t2           = np.array([float(r["t2"]) for r in self.data])
        delta_slave  = t2[self.delta:] - t2[:-self.delta]
        delta_master = t1[self.delta:] - t1[:-self.delta]
        y_est        = (delta_slave - delta_master) / delta_master

        for i,r in enumerate(self.data[self.delta:]):
            r["y_est"] = y_est[i]

    def optimize(self):
        """Optimize observation interval for minimum MSE

        """

        log_min_window = 1
        log_max_window = int(np.log2(len(self.data) / 2))
        log_window_len = np.arange(log_min_window, log_max_window + 1, 1)
        window_len     = 2**log_window_len
        max_window_len = window_len[-1]
        n_samples      = len(self.data) - max_window_len
        # NOTE: n_samples is a number of samples that is guaranteed to be
        # available for all window lengths to be evaluated

        logger.info("Optimize observation window" %())
        logger.info("Try from N = %d to N = %d" %(
            2**log_min_window, 2**log_max_window))

        mmse  = np.inf
        N_opt = 0

        for N in window_len:
            for r in self.data:
                r.pop("y_est", None)

            self.delta = N
            self.process()

            y_err = np.array([ 1e9*(r["y_est"] - r["rtc_y"])
                               for r in self.data[self.delta:]
                               if ("y_est" in r and "rtc_y" in r) ])
            y_mse = np.square(y_err[:n_samples]).mean()
            # Only use `n_samples` out of y_err. This way, all window lengths
            # are compared based on the same number of samples.

            if (y_mse < mmse):
                N_opt = N
                mmse  = y_mse

        logger.info("Minimum MSE: %f ppb" %(mmse))
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

        t1  = np.array([float(r["t1"]) for r in self.data])
        x   = np.array([r["x"] for r in self.data])
        dx  = x[self.delta:] - x[:-self.delta]
        dt1 = t1[self.delta:] - t1[:-self.delta]
        y   = dx / dt1

        for i,r in enumerate(self.data[self.delta:]):
            # Add to dataset
            r["rtc_y"] = y[i]

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


