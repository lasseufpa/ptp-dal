import unittest
import copy
import numpy as np
from ptp.frequency import *


immutable_data = [
    {"t1": 0,  "t2": 18, "t3": 32, "t4": 48},
    {"t1": 10, "t2": 26, "t3": 40, "t4": 52},
    {"t1": 20, "t2": 38, "t3": 50, "t4": 62},
    {"t1": 30, "t2": 50, "t3": 60, "t4": 75},
    {"t1": 40, "t2": 52, "t3": 70, "t4": 79}
]


class TestFrequency(unittest.TestCase):
    def setUp(self):
        self.data = copy.deepcopy(immutable_data)
        for elem in self.data:
            assert(elem["t4"] > elem["t1"])
            assert(elem["t3"] > elem["t2"])
            t21           = elem["t2"] - elem["t1"]
            t43           = elem["t4"] - elem["t3"]
            elem["x_est"] = (t21 - t43) / 2
            # Expected values:
            # [1, 2, 3, 2.5, 1.5]

    def test_one_way_foffset_est(self):
        """Test unbiased frequency offset estimates based on t1 and t2 only"""
        N = 3
        freq_estimator = Estimator(self.data, delta=N)
        freq_estimator.process(strategy="one-way")

        # We want a window spanning N sample intervals. Hence, we compute the
        # frequency offset between the extremes of windows containing N+1
        # samples. The first N samples should not contain any estimate. The
        # first estimate comes on sample index N (the N+1-th sample).
        assert(all([not ("y_est" in r) for r in self.data[:N]]))
        assert("y_est" in self.data[N])

        # Check estimates:
        y_est          = [r["y_est"] for r in self.data if "y_est" in r]
        expected_y_est = [((50-18) - 30)/30,
                          ((52-26) - 30)/30]
        self.assertListEqual(y_est, expected_y_est)

    def test_reversed_one_way_foffset_est(self):
        """Test unbiased frequency offset estimates based on t3 and t4 only"""
        N = 3
        freq_estimator = Estimator(self.data, delta=N)
        freq_estimator.process(strategy="one-way-reversed")

        # Refer to the comments on test_one_way_foffset_est()
        assert(all([not ("y_est" in r) for r in self.data[:N]]))
        assert("y_est" in self.data[N])

        # Check estimates:
        y_est          = [r["y_est"] for r in self.data if "y_est" in r]
        expected_y_est = [((60-32) - (75-48))/(75-48),
                          ((70-40) - (79-52))/(79-52)]
        self.assertListEqual(y_est, expected_y_est)

    def test_two_way_foffset_est(self):
        """Test unbiased frequency offset estimates based on t1/t2/t3/t4"""
        N = 3
        freq_estimator = Estimator(self.data, delta=N)
        freq_estimator.process(strategy="two-way")

        # Refer to the comments on test_one_way_foffset_est()
        assert(all([not ("y_est" in r) for r in self.data[:N]]))
        assert("y_est" in self.data[N])

        # Check estimates:
        y_est          = [r["y_est"] for r in self.data if "y_est" in r]
        expected_y_est = [(2.5-1)/30, (1.5-2)/30]
        self.assertListEqual(y_est, expected_y_est)

    def test_toffset_drift_est(self):
        """Test time offset drift estimation"""
        N = 3
        freq_estimator = Estimator(self.data, delta=N)
        freq_estimator.process(strategy="two-way")
        freq_estimator.estimate_drift()

        # The drift estimates are only added to the dataset entries that contain
        # a corresponding frequency offset estimate.
        assert(all([not ("drift" in r) for r in self.data[:N]]))
        assert("drift" in self.data[N])

        # Check estimates. The drift estimate at the n-th sample is given by the
        # n-th frequency offset estimate multiplied by the interval between
        # sample n-1 and sample n.
        drift_est      = [r["drift"] for r in self.data if "drift" in r]
        expected_y_est = [(2.5-1)/30, (1.5-2)/30]
        expected_drift = [10*x for x in expected_y_est]
        self.assertListEqual(drift_est, expected_drift)

