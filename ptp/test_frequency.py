import copy
import unittest

import numpy as np

from ptp.frequency import Estimator

immutable_data = [{
    "t1": 0,
    "t2": 18,
    "t3": 32,
    "t4": 48,
    "x": 1.5
}, {
    "t1": 10,
    "t2": 26,
    "t3": 40,
    "t4": 52,
    "x": 2.2
}, {
    "t1": 20,
    "t2": 38,
    "t3": 50,
    "t4": 62,
    "x": 3.2
}, {
    "t1": 30,
    "t2": 50,
    "t3": 60,
    "t4": 75,
    "x": 2.8
}, {
    "t1": 40,
    "t2": 52,
    "t3": 70,
    "t4": 79,
    "x": 1.8
}]


class TestFrequency(unittest.TestCase):
    def setUp(self):
        self.data = copy.deepcopy(immutable_data)
        for elem in self.data:
            assert (elem["t4"] > elem["t1"])
            assert (elem["t3"] > elem["t2"])
            t21 = elem["t2"] - elem["t1"]
            t43 = elem["t4"] - elem["t3"]
            elem["x_est"] = (t21 - t43) / 2
            # Expected values:
            # [1, 2, 3, 2.5, 1.5]

    def _estimate_foffset(self, strategy, N=3):
        self.estimator = Estimator(self.data, delta=N)
        self.estimator.process(strategy=strategy)

        # We want a window spanning N sample intervals. Hence, we compute the
        # frequency offset between the extremes of windows containing N+1
        # samples. The first N samples should not contain any estimate. The
        # first estimate comes on sample index N (the N+1-th sample).
        assert (all([not ("y_est" in r) for r in self.data[:N]]))
        assert ("y_est" in self.data[N])

    def _estimate_drift(self, N=3):
        self.estimator.estimate_drift()

        # The drift estimates are only added to the dataset entries that
        # contain a corresponding frequency offset estimate.
        assert (all([not ("drift" in r) for r in self.data[:N]]))
        assert ("drift" in self.data[N])

    def test_one_way_foffset_est(self):
        """Test unbiased frequency offset estimates based on t1 and t2 only"""
        self._estimate_foffset(strategy="one-way")

        # Check estimates:
        y_est = [r["y_est"] for r in self.data if "y_est" in r]
        expected_y_est = [((50 - 18) - 30) / 30, ((52 - 26) - 30) / 30]
        self.assertListEqual(y_est, expected_y_est)

    def test_reversed_one_way_foffset_est(self):
        """Test unbiased frequency offset estimates based on t3 and t4 only"""
        self._estimate_foffset(strategy="one-way-reversed")

        # Check estimates:
        y_est = [r["y_est"] for r in self.data if "y_est" in r]
        expected_y_est = [((60 - 32) - (75 - 48)) / (75 - 48),
                          ((70 - 40) - (79 - 52)) / (79 - 52)]
        self.assertListEqual(y_est, expected_y_est)

    def test_two_way_foffset_est(self):
        """Test unbiased frequency offset estimates based on t1/t2/t3/t4"""
        self._estimate_foffset(strategy="two-way")

        # Check estimates:
        y_est = [r["y_est"] for r in self.data if "y_est" in r]
        expected_y_est = [(2.5 - 1) / 30, (1.5 - 2) / 30]
        self.assertListEqual(y_est, expected_y_est)

    def test_toffset_drift_est(self):
        """Test time offset drift estimation"""
        self._estimate_foffset(strategy="two-way")
        self._estimate_drift()

        # Check estimates. The drift estimate at the n-th sample is given by
        # the n-th frequency offset estimate multiplied by the interval between
        # sample n-1 and sample n.
        drift_est = [r["drift"] for r in self.data if "drift" in r]
        expected_y_est = [(2.5 - 1) / 30, (1.5 - 2) / 30]
        expected_drift = [10 * x for x in expected_y_est]
        self.assertListEqual(drift_est, expected_drift)

    def test_drift_err_eval(self):
        """Test the drift estimation error evaluation"""
        self._estimate_foffset(strategy="two-way")
        self._estimate_drift()

        expected_drift = np.array([(2.5 - 1) / 3, (1.5 - 2) / 3])
        true_drift = np.array([-0.4, -1])
        err = expected_drift - true_drift
        norm_drift_err = err.cumsum() / true_drift.cumsum()
        # Error should be equal to 0.9 and (2.5/3)
        # Cumulative error should be 0.9 and (0.9 + 2.5/3)
        # True cumulative time offset drift is [-0.4, -1.4]
        # Normalized cumulative error is [0.9/0.4, (0.9 + 2.5/3)/1.4]

        # MSE loss function
        drift_err = self.estimator._eval_drift_err("mse", "instantaneous")
        cum_drift_err = self.estimator._eval_drift_err("mse", "cumulative")
        self.assertAlmostEqual(drift_err, np.square(err).mean())
        self.assertAlmostEqual(cum_drift_err, np.square(norm_drift_err).mean())

        # max|error| loss function
        drift_err = self.estimator._eval_drift_err("max-error",
                                                   "instantaneous")
        cum_drift_err = self.estimator._eval_drift_err("max-error",
                                                       "cumulative")
        self.assertAlmostEqual(drift_err, 0.9)
        self.assertAlmostEqual(cum_drift_err, 0.9 / 0.4)

        # Test restricted set of samples on evaluation. More specifically, test
        # the error evaluation based on the last sample only. In this case, the
        # max|error| and MSE are equal to the last error value and its square
        # value, respectively.
        last_err = (2.5 / 3)
        last_norm_cum_err = (0.9 + 2.5 / 3) / 1.4

        # MSE loss function
        drift_err = self.estimator._eval_drift_err("mse",
                                                   "instantaneous",
                                                   n_samples=1)
        cum_drift_err = self.estimator._eval_drift_err("mse",
                                                       "cumulative",
                                                       n_samples=1)
        self.assertAlmostEqual(drift_err, last_err**2)
        self.assertAlmostEqual(cum_drift_err, last_norm_cum_err**2)

        # max|error| loss function
        drift_err = self.estimator._eval_drift_err("max-error",
                                                   "instantaneous",
                                                   n_samples=1)
        cum_drift_err = self.estimator._eval_drift_err("max-error",
                                                       "cumulative",
                                                       n_samples=1)
        self.assertAlmostEqual(drift_err, last_err)
        self.assertAlmostEqual(cum_drift_err, last_norm_cum_err)
