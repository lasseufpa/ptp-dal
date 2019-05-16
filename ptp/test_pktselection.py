import unittest
from ptp.pktselection import *

data = [{"x_est": 6 , "d_est": 2},
        {"x_est": 6, "d_est": 1},
        {"x_est": 15,"d_est": 3},
        {"x_est": 17, "d_est": 2},
        {"x_est": 56, "d_est": 3}]

class TestPktSelection(unittest.TestCase):

    def test_sample_mean(self):
        N    = 2
        pkts = PktSelection(N, data)
        pkts.process('average', avg_impl="normal")
        x_est_avg = [r["x_pkts_average"] for r in data if "x_pkts_average" in r]

        # Check values
        self.assertEqual(x_est_avg[0], 6)
        self.assertEqual(x_est_avg[1], 10.5)
        self.assertEqual(x_est_avg[2], 16)
        self.assertEqual(x_est_avg[3], 36.5)

    def test_sample_recursive(self):
        for r in data:
            r.pop("x_pkts_average", None)
        N    = 3
        pkts = PktSelection(N, data)
        pkts.process('average', avg_impl="recursive")
        x_est_avg = [r["x_pkts_average"] for r in data if "x_pkts_average" in r]

        # Check values
        self.assertEqual(x_est_avg[0], 6/3)
        self.assertEqual(x_est_avg[1], 12/3)
        self.assertEqual(x_est_avg[2], 27/3)
        self.assertEqual(x_est_avg[3], 38/3)
        self.assertEqual(x_est_avg[4], 88/3)

    def test_sample_median(self):
        N    = 3
        pkts = PktSelection(N, data)
        pkts.process('median')
        x_est_median = [r["x_pkts_median"] for r in data if "x_pkts_median" in r]

        # Check values
        self.assertEqual(x_est_median[0], 6)
        self.assertEqual(x_est_median[1], 15)
        self.assertEqual(x_est_median[2], 17)

    def test_sample_min(self):
        N    = 3
        pkts = PktSelection(N, data)
        pkts.process('min')
        x_est_min = [r["x_pkts_min"] for r in data if "x_pkts_min" in r]

        # Check values
        self.assertEqual(x_est_min[0], 6)
        self.assertEqual(x_est_min[1], 6)
        self.assertEqual(x_est_min[2], 17)

