import copy
import unittest

from ptp.pktselection import *

immutable_data = [{
    "x_est": 6,
    "d_est": 2,
    "drift": 1,
    "t1": 0,
    "t2": 18,
    "t3": 30,
    "t4": 38
}, {
    "x_est": 6,
    "d_est": 1,
    "drift": 2,
    "t1": 0,
    "t2": 12,
    "t3": 30,
    "t4": 42
}, {
    "x_est": 15,
    "d_est": 3,
    "drift": 5,
    "t1": 0,
    "t2": 16,
    "t3": 30,
    "t4": 45
}, {
    "x_est": 17,
    "d_est": 2,
    "drift": 3,
    "t1": 0,
    "t2": 19,
    "t3": 30,
    "t4": 55
}, {
    "x_est": 56,
    "d_est": 3,
    "drift": 4,
    "t1": 0,
    "t2": 12,
    "t3": 30,
    "t4": 41
}]

# Derived vectors
#
# x_est                      = [6, 6, 15, 17, 56]
# t21                        = [18, 12, 16, 19, 12]
# t43                        = [8, 12, 15, 25, 11]
# drift                      = [1, 2, 5, 3, 4]
# cum_drift                  = [1, 3, 8, 11, 15]
# t21' = t21 - cum_drift     = [17, 9, 8, 8, -3]
# t43' = t43 + cum_drift     = [9, 15, 23, 36, 26]
# x_est' = x_est - cum_drift = [5, 3, 7, 6, 41]


class TestPktSelection(unittest.TestCase):
    def _run_sample_avg(self,
                        drift_comp=False,
                        vectorize=False,
                        recursive=False,
                        batch=False):
        """Window-based sample-average test runner"""
        data = copy.deepcopy(immutable_data)
        N = 2
        pkts = PktSelection(N, data)
        pkts.process('avg',
                     drift_comp=drift_comp,
                     vectorize=vectorize,
                     recursive=recursive,
                     batch=batch)
        x_est_avg = [r["x_pkts_avg"] for r in data if "x_pkts_avg" in r]

        # Check values
        if (drift_comp):
            # avg(x_est') in N      = [4, 5, 6.5, 23.5]
            # after re-adding drift = [7, 13, 17.5, 38.5]
            self.assertEqual(x_est_avg[0], 7)
            self.assertEqual(x_est_avg[1], 13)
            self.assertEqual(x_est_avg[2], 17.5)
            self.assertEqual(x_est_avg[3], 38.5)
        else:
            # avg(x_est) in N = [6, 10.5, 16, 36.5]
            self.assertEqual(x_est_avg[0], 6)
            self.assertEqual(x_est_avg[1], 10.5)
            self.assertEqual(x_est_avg[2], 16)
            self.assertEqual(x_est_avg[3], 36.5)

    def test_sample_avg_normal(self):
        """Window-based sample-average"""
        for drift_comp in [True, False]:
            self._run_sample_avg(drift_comp=drift_comp)

    def test_sample_avg_recursive(self):
        """Recursive sample-average"""
        for drift_comp in [True, False]:
            self._run_sample_avg(drift_comp=drift_comp, recursive=True)

    def test_sample_avg_normal_vec(self):
        """Vectorized sample-average with and without batch processing"""
        for drift_comp in [True, False]:
            for batch in [True, False]:
                self._run_sample_avg(drift_comp=drift_comp,
                                     vectorize=True,
                                     batch=batch)

    def _run_sample_median(self,
                           drift_comp=False,
                           vectorize=False,
                           batch=False):
        """Sample-median test runner"""
        data = copy.deepcopy(immutable_data)
        N = 3
        pkts = PktSelection(N, data)
        pkts.process('median',
                     drift_comp=drift_comp,
                     vectorize=vectorize,
                     batch=batch)
        x_est_median = [r["x_pkts_median"] for r in data if \
                         "x_pkts_median" in r]

        # Check values
        if (drift_comp):
            # median(t21') in N     = [9, 8, 8]
            # median(t43') in N     = [15, 23, 26]
            # sample-median result  = [-3, -7.5, -9]
            # after re-adding drift = [5, 3.5, 6]
            self.assertEqual(x_est_median[0], 5)
            self.assertEqual(x_est_median[1], 3.5)
            self.assertEqual(x_est_median[2], 6)
        else:
            # median(t21) in N = [16, 16, 16]
            # median(t43) in N = [12, 15, 15]
            self.assertEqual(x_est_median[0], (16 - 12) / 2)
            self.assertEqual(x_est_median[1], (16 - 15) / 2)
            self.assertEqual(x_est_median[2], (16 - 15) / 2)

    def test_sample_median(self):
        """Non-vectorized sample-median"""
        for drift_comp in [True, False]:
            self._run_sample_median(drift_comp=drift_comp)

    def test_sample_median_vec(self):
        """Vectorized sample-median with and without batch processing"""
        for drift_comp in [True, False]:
            for batch in [True, False]:
                self._run_sample_median(drift_comp=drift_comp,
                                        vectorize=True,
                                        batch=batch)

    def _run_sample_min(self,
                        drift_comp=False,
                        vectorize=False,
                        recursive=False,
                        batch=False):
        """Sample-minimum test runner"""
        data = copy.deepcopy(immutable_data)
        N = 3
        pkts = PktSelection(N, data)
        pkts.process('min',
                     drift_comp=drift_comp,
                     vectorize=vectorize,
                     recursive=recursive,
                     batch=batch)
        x_est_min = [r["x_pkts_min"] for r in data if "x_pkts_min" in r]

        # Check values
        if (drift_comp):
            # min(t21') in N        = [8, 8, -3]
            # min(t43') in N        = [9, 15, 23]
            # sample-min result     = [-0.5, -3.5, -13]
            # after re-adding drift = [7.5, 7.5, 2]
            self.assertEqual(x_est_min[0], 7.5)
            self.assertEqual(x_est_min[1], 7.5)
            self.assertEqual(x_est_min[2], 2)
        else:
            # min(t21) in N = [12, 12, 12]
            # min(t43) in N = [8, 12, 11]
            self.assertEqual(x_est_min[0], 2)
            self.assertEqual(x_est_min[1], 0)
            self.assertEqual(x_est_min[2], 0.5)

    def test_sample_min(self):
        """Non-vectorized and non-recursive sample-minimum"""
        for drift_comp in [True, False]:
            self._run_sample_min(drift_comp=drift_comp)

    def test_sample_min_recursive(self):
        """Recursive sample-minimum"""
        for drift_comp in [True, False]:
            self._run_sample_min(drift_comp=drift_comp, recursive=True)

    def test_sample_min_vec(self):
        """Vectorized sample-minimum with and without batch processing"""
        for drift_comp in [True, False]:
            for batch in [True, False]:
                self._run_sample_min(drift_comp=drift_comp,
                                     vectorize=True,
                                     batch=batch)

    def _run_sample_max(self,
                        drift_comp=False,
                        vectorize=False,
                        recursive=False,
                        batch=False):
        """Sample-maximum test runner"""
        data = copy.deepcopy(immutable_data)
        N = 3
        pkts = PktSelection(N, data)
        pkts.process('max',
                     drift_comp=drift_comp,
                     vectorize=vectorize,
                     recursive=recursive,
                     batch=batch)
        x_est_max = [r["x_pkts_max"] for r in data if "x_pkts_max" in r]

        # Check values
        if (drift_comp):
            # max(t21') in N        = [17, 9, 8]
            # max(t43') in N        = [23, 36, 36]
            # sample-max result     = [-3, -13.5, -14]
            # after re-adding drift = [5, -2.5, 1]
            self.assertEqual(x_est_max[0], 5)
            self.assertEqual(x_est_max[1], -2.5)
            self.assertEqual(x_est_max[2], 1)
        else:
            # max(t21) in N = [18, 19, 19]
            # max(t43) in N = [15, 25, 25]
            self.assertEqual(x_est_max[0], 1.5)
            self.assertEqual(x_est_max[1], -3)
            self.assertEqual(x_est_max[2], -3)

    def test_sample_max(self):
        """Non-vectorized sample-maximum"""
        for drift_comp in [True, False]:
            self._run_sample_max(drift_comp=drift_comp)

    def test_sample_max_recursive(self):
        """Recursive sample-maximum"""
        for drift_comp in [True, False]:
            self._run_sample_max(drift_comp=drift_comp, recursive=True)

    def test_sample_max_vec(self):
        """Vectorized sample-maximum with and without batch processing"""
        for drift_comp in [True, False]:
            for batch in [True, False]:
                self._run_sample_max(drift_comp=drift_comp,
                                     vectorize=True,
                                     batch=batch)

    def _run_sample_mode(self,
                         drift_comp=False,
                         vectorize=False,
                         recursive=False,
                         batch=False):
        """Sample-mode test runner"""
        data = copy.deepcopy(immutable_data)
        N = 3
        pkts = PktSelection(N, data)
        pkts.process('mode',
                     drift_comp=drift_comp,
                     vectorize=vectorize,
                     recursive=recursive,
                     batch=batch)
        x_est_mode = [r["x_pkts_mode"] for r in data if "x_pkts_mode" in r]

        # Check results
        if (drift_comp):
            # Timestamp differences:
            # t2' - t1': [[17. 9. 8.]
            #             [9. 8. 8.]
            #             [8. 8. -3.]]
            # t4' - t3': [[ 9. 15. 23.]
            #             [15. 23. 36.]
            #             [23. 36. 26.]]
            #
            # Quantized timestamp differences (quantum = 10 ns):
            # NOTE: 0.5 is rounded to 1
            # t2' - t1': [[2. 1. 1.]
            #             [1. 1. 1.]
            #             [1. 1. 0.]]
            # t4' - t3': [[1. 2. 2.]
            #             [2. 2. 4.]
            #             [2. 4. 3.]]
            #
            # Mode in each quantized window:
            # NOTE: if there isn't a mode, the lowest number wins
            # t2' - t1': [[1.]
            #             [1.]
            #             [1.]]
            # t4' - t3': [[2.]
            #             [2.]
            #             [2.]]
            #
            # After dequantization
            # t2' - t1': [[10.]
            #             [10.]
            #             [10.]]
            # t4' - t3': [[20.]
            #             [20.]
            #             [20.]]
            #
            # Sample-mode results   = [-5, -5, -5]
            # after re-adding drift = [3, 6, 10]
            self.assertEqual(x_est_mode[0], 3)
            self.assertEqual(x_est_mode[1], 6)
            self.assertEqual(x_est_mode[2], 10)
        else:
            # Timestamp differences:
            # t2 - t1: [[18. 12. 16.]
            #           [12. 16. 19.]
            #           [16. 19. 12.]]
            # t4 - t3: [[ 8. 12. 15.]
            #           [12. 15. 25.]
            #           [15. 25. 11.]]
            #
            # Quantized timestamp differences (quantum = 10 ns):
            # t2 - t1: [[2. 1. 2.]
            #           [1. 2. 2.]
            #           [2. 2. 1.]]
            # t4 - t3: [[1. 1. 2.]
            #           [1. 2. 2.]
            #           [2. 2. 1.]]
            #
            # Mode in each quantized window:
            # t2 - t1: [[2.]
            #           [2.]
            #           [2.]]
            # t4 - t3: [[1.]
            #           [2.]
            #           [2.]]
            #
            # After dequantization
            # t2 - t1: [[20.]
            #           [20.]
            #           [20.]]
            # t4 - t3: [[10.]
            #           [20.]
            #           [20.]]
            #
            # Expected results
            self.assertEqual(x_est_mode[0], 5.0)
            self.assertEqual(x_est_mode[1], 0)
            self.assertEqual(x_est_mode[2], 0)

    def test_sample_mode_vec(self):
        """Vectorized sample-mode with and without batch processing"""
        for drift_comp in [True, False]:
            for batch in [True, False]:
                self._run_sample_mode(drift_comp=drift_comp,
                                      vectorize=True,
                                      batch=batch)

    def test_sample_mode_recursive(self):
        """Recursive sample-mode"""
        for drift_comp in [True, False]:
            self._run_sample_mode(drift_comp=drift_comp, recursive=True)
