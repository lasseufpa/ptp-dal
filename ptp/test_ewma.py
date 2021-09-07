import unittest

import numpy as np

from ptp.ewma import *


class TestEwma(unittest.TestCase):
    def test_direct_alpha_beta(self):
        """Test alpha and beta set directly"""
        alpha = 0.1
        beta = 1 - alpha
        ewma = Ewma(beta=beta)
        self.assertAlmostEqual(ewma.alpha, alpha)
        self.assertAlmostEqual(ewma.beta, beta)

    def test_indirect_alpha_beta(self):
        """Test EWMA's alpha and beta settings based on the target window"""
        N = 10
        ewma = Ewma()
        ewma.set_equivalent_window(N)
        self.assertEqual(ewma.alpha, 1 / N)
        self.assertEqual(ewma.beta, (1 - 1 / N))

    def test_avg_no_bias_corr(self):
        """Test EWMA without bias correction"""
        x_vec = [1, 2, 3]
        N = 3
        ewma = Ewma(bias_corr=False)
        ewma.set_equivalent_window(N)

        for x in x_vec:
            avg = ewma.step(x)

        # Expected:
        a = 1 / N  # alpha
        avg_exp = 3 * a + (1 - a) * (2 * a + (1 - a) * (1 * a))
        self.assertEqual(avg, avg_exp)

    def test_avg_w_bias_corr(self):
        """Test EWMA with bias correction"""
        x_vec = [1, 2, 3]
        N = 3
        ewma = Ewma(bias_corr=True)
        ewma.set_equivalent_window(N)

        for x in x_vec:
            avg = ewma.step(x)

        # Expected:
        a = 1 / N  # alpha
        b = 1 - a  # beta
        avg_exp = 3 * a + (1 - a) * (2 * a + (1 - a) * (1 * a))
        bias_corr = 1 / (1 - (b**3))
        self.assertEqual(avg, avg_exp * bias_corr)
