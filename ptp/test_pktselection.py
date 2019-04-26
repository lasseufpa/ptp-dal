import unittest
from ptp.pktselection import *

data = [{"x_est": 6 , "d_est": 2},
        {"x_est": 6, "d_est": 1},
        {"x_est": 15,"d_est": 3},
        {"x_est": 17, "d_est": 2},
        {"x_est": 56, "d_est": 3}]

class TestPktSelection(unittest.TestCase):

    def test_sample_mean(self):
        pkts = PktSelection(2, data)
        pkts.process('mean')
        x_est_mean = [r["x_pkts_mean"] for r in data if "x_pkts_mean" in r]

        # Check values
        self.assertEqual(x_est_mean[0], 6)
        self.assertEqual(x_est_mean[1], 10.5)
        self.assertEqual(x_est_mean[2], 16)
        self.assertEqual(x_est_mean[3], 36.5)

    def test_sample_median(self):
        pkts = PktSelection(3, data)
        pkts.process('median')
        x_est_median = [r["x_pkts_median"] for r in data if "x_pkts_median" in r]

        # Check values
        self.assertEqual(x_est_median[0], 6)
        self.assertEqual(x_est_median[1], 15)
        self.assertEqual(x_est_median[2], 17)

    def test_sample_min(self):
        pkts= PktSelection(3, data)
        pkts.process('min')
        x_est_min = [r["x_pkts_min"] for r in data if "x_pkts_min" in r]

        # Check values
        self.assertEqual(x_est_min[0], 6)
        self.assertEqual(x_est_min[1], 6)
        self.assertEqual(x_est_min[2], 17)

