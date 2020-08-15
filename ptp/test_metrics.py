import unittest
import numpy as np
from ptp.metrics import *


class TestMetrics(unittest.TestCase):
    def test_max_te(self):
        data = [
            {"x_est": 8,  "x" : 15},
            {"x_est": 10, "x" : 4},
            {"x_est": 6,  "x" : 9},
            {"x_est": 12, "x" : 8},
            {"x_est": 14, "x" : 9}
        ]
        analyser = Analyser(data)

        N  = 2
        te = np.array([r["x_est"] - r["x"] for r in data])

        max_te = analyser.max_te(te, N)

        # Expected TE: [-7, 6, -3, 4, 5]
        expected_max_te = [7, 4]
        # NOTE: max|TE| is computed with non-overlapping windows. If the last
        # window is incomplete, the samples are ignored.

        self.assertListEqual(max_te.tolist(), expected_max_te)

