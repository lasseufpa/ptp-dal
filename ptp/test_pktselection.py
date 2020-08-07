import unittest
import copy
from ptp.pktselection import *

immutable_data = [
    {"x_est": 6 , "d_est": 2, "t1": 0, "t2": 18, "t3": 30, "t4": 38},
    {"x_est": 6,  "d_est": 1, "t1": 0, "t2": 12, "t3": 30, "t4": 42},
    {"x_est": 15, "d_est": 3, "t1": 0, "t2": 16, "t3": 30, "t4": 45},
    {"x_est": 17, "d_est": 2, "t1": 0, "t2": 19, "t3": 30, "t4": 55},
    {"x_est": 56, "d_est": 3, "t1": 0, "t2": 12, "t3": 30, "t4": 41}
]

class TestPktSelection(unittest.TestCase):

    def test_sample_avg_normal(self):
        data = copy.deepcopy(immutable_data)
        N    = 2
        pkts = PktSelection(N, data)
        pkts.process('avg-normal', drift_comp = False, vectorize = False)
        x_est_avg = [r["x_pkts_avg_normal"] for r in data if "x_pkts_avg_normal" in r]

        # Check values
        self.assertEqual(x_est_avg[0], 6)
        self.assertEqual(x_est_avg[1], 10.5)
        self.assertEqual(x_est_avg[2], 16)
        self.assertEqual(x_est_avg[3], 36.5)

    def test_sample_avg_normal_vec(self):
        # Run vectorized processing with and without batch processing
        for batch in [True, False]:
            data = copy.deepcopy(immutable_data)
            N    = 2
            pkts = PktSelection(N, data)
            pkts.process('avg-normal', drift_comp=False, vectorize=True,
                         batch=batch, batch_size=3)
            x_est_avg = [r["x_pkts_avg_normal"] for r in data if "x_pkts_avg_normal" in r]

            # Check values
            self.assertEqual(x_est_avg[0], 6)
            self.assertEqual(x_est_avg[1], 10.5)
            self.assertEqual(x_est_avg[2], 16)
            self.assertEqual(x_est_avg[3], 36.5)

    def test_sample_avg_recursive(self):
        data = copy.deepcopy(immutable_data)
        N    = 3
        pkts = PktSelection(N, data)
        pkts.process('avg-recursive', drift_comp=False, vectorize=False)
        x_est_avg = [r["x_pkts_avg_recursive"] for r in data if \
                     "x_pkts_avg_recursive" in r]

        # Check values
        self.assertEqual(x_est_avg[0], 38/3)
        self.assertEqual(x_est_avg[1], 88/3)
        # NOTE: after skipping the transient of N=3 samples, only two samples
        # are expected.

    def test_sample_avg_recursive_vec(self):
        # Run vectorized processing with and without batch processing
        for batch in [True, False]:
            data = copy.deepcopy(immutable_data)
            N    = 3
            pkts = PktSelection(N, data)
            pkts.process('avg-recursive', drift_comp=False, vectorize=True,
                         batch=batch, batch_size=3)
            x_est_avg = [r["x_pkts_avg_recursive"] for r in data if \
                         "x_pkts_avg_recursive" in r]

            # Check values after transient
            self.assertEqual(x_est_avg[0], 38/3)
            self.assertEqual(x_est_avg[1], 88/3)

    def test_sample_median(self):
        data = copy.deepcopy(immutable_data)
        N    = 3
        pkts = PktSelection(N, data)
        pkts.process('median', drift_comp=False, vectorize=False)
        x_est_median = [r["x_pkts_median"] for r in data if \
                         "x_pkts_median" in r]

        # Check values
        self.assertEqual(x_est_median[0], (16 - 12)/2)
        self.assertEqual(x_est_median[1], (16 - 15)/2)
        self.assertEqual(x_est_median[2], (16 - 15)/2)

    def test_sample_median_vec(self):
        # Run vectorized processing with and without batch processing
        for batch in [True, False]:
            data = copy.deepcopy(immutable_data)
            N    = 3
            pkts = PktSelection(N, data)
            pkts.process('median', drift_comp=False, vectorize=True,
                         batch=batch, batch_size=3)
            x_est_median = [r["x_pkts_median"] for r in data if \
                             "x_pkts_median" in r]

            # Check values
            self.assertEqual(x_est_median[0], (16 - 12)/2)
            self.assertEqual(x_est_median[1], (16 - 15)/2)
            self.assertEqual(x_est_median[2], (16 - 15)/2)

    def test_sample_min(self):
        data = copy.deepcopy(immutable_data)
        N    = 3
        pkts = PktSelection(N, data)
        pkts.process('min', drift_comp=False, vectorize=False)
        x_est_min = [r["x_pkts_min"] for r in data if "x_pkts_min" in r]

        # Check values
        self.assertEqual(x_est_min[0], 2)
        self.assertEqual(x_est_min[1], 0)
        self.assertEqual(x_est_min[2], 0.5)

    def test_sample_min_vec(self):
        # Run vectorized processing with and without batch processing
        for batch in [True, False]:
            data = copy.deepcopy(immutable_data)
            N    = 3
            pkts = PktSelection(N, data)
            pkts.process('min', drift_comp=False, vectorize=True,
                         batch=batch, batch_size=3)
            x_est_min = [r["x_pkts_min"] for r in data if "x_pkts_min" in r]

            # Check values
            self.assertEqual(x_est_min[0], 2)
            self.assertEqual(x_est_min[1], 0)
            self.assertEqual(x_est_min[2], 0.5)

    def test_sample_max(self):
        data = copy.deepcopy(immutable_data)
        N    = 3
        pkts = PktSelection(N, data)
        pkts.process('max', drift_comp=False, vectorize=False)
        x_est_max = [r["x_pkts_max"] for r in data if "x_pkts_max" in r]

        # Check values
        self.assertEqual(x_est_max[0], 1.5)
        self.assertEqual(x_est_max[1], -3)
        self.assertEqual(x_est_max[2], -3)

    def test_sample_max_vec(self):
        # Run vectorized processing with and without batch processing
        for batch in [True, False]:
            data = copy.deepcopy(immutable_data)
            N    = 3
            pkts = PktSelection(N, data)
            pkts.process('max', drift_comp=False, vectorize=True,
                         batch=batch, batch_size=3)
            x_est_max = [r["x_pkts_max"] for r in data if "x_pkts_max" in r]

            # Check values
            self.assertEqual(x_est_max[0], 1.5)
            self.assertEqual(x_est_max[1], -3)
            self.assertEqual(x_est_max[2], -3)

