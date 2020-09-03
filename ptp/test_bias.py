import unittest
import copy
from ptp.bias import *

immutable_data = [
    {"x_est": 20, "d": 20, "d_bw": 10},
    {"x_est": 25, "d": 25, "d_bw": 14},
    {"x_est": 15, "d": 26, "d_bw": 12},
    {"x_est": 17, "d": 22, "d_bw": 13},
    {"x_est": 30, "d": 28, "d_bw": 15}
]

class TestBias(unittest.TestCase):
    def setUp(self):
        data = copy.deepcopy(immutable_data)
        self.bias = Bias(data)

    def test_true_bias_calc(self):
        """Test the calculation of the true bias using the true delays"""
        expected_asymmetry = {
            'avg'    : (10 + 11 + 14 + 9 + 13)/5/2,
            'min'    : (20 - 10)/2,
            'max'    : (28 - 15)/2,
            'median' : (25 - 13)/2,
            'mode'   : (20 - 10)/2
        }
        # NOTE: the mode computation returns the lowest number in the set when
        # there are no repeated numbers (20 for "d" and 10 for "d_bw").

        bias_est = {}
        for metric in ['avg', 'min', 'max', 'median', 'mode']:
            bias_est[metric] = self.bias.calc_true_asymmetry(metric=metric)
            self.assertAlmostEqual(expected_asymmetry[metric], bias_est[metric])

    def test_bias_corrections(self):
        """Test bias correction"""
        self.bias.compensate(corr = 5, toffset_key='x_est')
        x_est          = [r['x_est'] for r in self.bias.data]
        expected_x_est = [15, 20, 10, 12, 25]
        self.assertEqual(x_est, expected_x_est)
