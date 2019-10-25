"""Outlier detection
"""
import logging
import numpy as np
logger = logging.getLogger(__name__)


class Outlier():
    def __init__(self, data):
        """Outliers detector

        Args:
            data : Array of objects with simulation data

        """
        self.data = data

    def _iqr(self, d_asym, c=1.5):
        """Interquartile Range (IQR) method

        The Interqurtile Range (IQR) is a measure of statistical dispersion,
        being equal to the difference between 75th and 25th percentiles.
        Therefore, the IQR method can be used to identify outliers by defining
        limits on the sample values that are a factor 'c' of the IQR below the
        25th percentile or above the 75th percentile. The default value for the
        factor 'c' is 1.5, but it can be increased to identify just values that
        are extreme outliers.

        Args:
            d_asym : Vector of delay asymmetries
            c      : Scalar value that define the limits on the sample values

        Returns:
            Vector with sample values defined as outliers

        """
        Q1, Q3      = np.percentile(d_asym, [25, 75])
        iqr         = Q3 - Q1
        lower_bound = Q1 - (iqr * c)
        upper_bound = Q3 + (iqr * c)
        outliers    = np.where((d_asym > upper_bound) | (d_asym < lower_bound))

        return np.squeeze(outliers)

    def process(self, c=1.5):
        """"Process the sample values

        Args:
            c : Scalar value that define the limits on the sample values

        """
        d_asym = np.array([r['asym'] for r in self.data])
        x      = np.array([r['idx'] for r in self.data])

        # Identify outliers
        outliers = self._iqr(d_asym)

        # Save results on global data records
        for out in outliers:
            self.data[out]['outlier'] = True
