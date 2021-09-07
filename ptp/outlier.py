"""Outlier detection
"""
import logging

import numpy as np

from ptp.mechanisms import DelayReqResp

logger = logging.getLogger(__name__)


class Outlier():
    def __init__(self, data):
        """Outliers detector

        Args:
            data : Array of objects with simulation data

        """
        self.data = data

    def _iqr(self, x, c=1.5):
        """Interquartile Range (IQR) method

        The Interqurtile Range (IQR) is a measure of statistical dispersion,
        being equal to the difference between 75th and 25th percentiles.
        Therefore, the IQR method can be used to identify outliers by defining
        limits on the sample values that are a factor 'c' of the IQR below the
        25th percentile or above the 75th percentile. The default value for the
        factor 'c' is 1.5, but it can be increased to identify just values that
        are extreme outliers.

        Args:
            x : Vector of samples whose outliers are to be detected
            c : Scalar value that define the limits on the sample values

        Returns:
            Vector with sample values defined as outliers

        """
        Q1, Q3 = np.percentile(x, [25, 75])
        iqr = Q3 - Q1
        lower_bound = Q1 - (iqr * c)
        upper_bound = Q3 + (iqr * c)
        outliers = np.where((x > upper_bound) | (x < lower_bound))[0]

        return outliers

    def _print_tstamps(self, idx):
        logger.info("{:^6d} | {:21s} | {:21s} | {:21s} | {:21s}".format(
            idx, str(self.data[idx]['t1']), str(self.data[idx]['t2']),
            str(self.data[idx]['t3']), str(self.data[idx]['t4'])))

    def _debug_outlier_context(self, idx):
        """Print some contextual info around the outlier

        Args:
            idx : Outlier index

        """

        # When
        t_out_ns = float(self.data[idx]["t1"] - self.data[0]["t1"])
        t_min = t_out_ns / (60 * 1e9)
        logger.info("---- Outlier at index {} - {} min".format(idx, t_min))

        # PTP Metrics
        logger.info("Metrics:")
        DelayReqResp.log_header(level=logging.INFO, logger=logger)
        DelayReqResp.log(self.data[idx - 1], logging.INFO, logger)
        DelayReqResp.log(self.data[idx], logging.INFO, logger)
        DelayReqResp.log(self.data[idx + 1], logging.INFO, logger)

        # PTP timestamps
        logger.info("Timestamps:")
        logger.info("{:6s} | {:21s} | {:21s} | {:21s} | {:21s}".format(
            "idx", "t1", "t2", "t3", "t4"))
        logger.info("----------------------------------------"
                    "----------------------------------------"
                    "----------------------")
        self._print_tstamps(idx - 1)
        self._print_tstamps(idx)
        self._print_tstamps(idx + 1)
        logger.info("----")

    def process(self, c=1.5):
        """"Process the sample values

        Args:
            c : Scalar value that define the limits on the sample values

        """
        d_asym = np.array([r['asym'] for r in self.data])

        # Identify outliers
        outliers = self._iqr(d_asym)

        logger.info("Found %d outliers" % (len(outliers)))

        # Save results on global data records
        for out in outliers:
            self.data[out]['outlier'] = True

            if ((logger.root.level == logging.INFO) and (out > 1)
                    and (out < (len(self.data) - 1))):
                self._debug_outlier_context(out)
