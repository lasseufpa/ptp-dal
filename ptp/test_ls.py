import unittest
import numpy as np
from ptp.ls import *

data = [{"x_est": 6 , "t1": 0,     "t2": 0},
        {"x_est": 8,  "t1": 250e6, "t2": 250e6},
        {"x_est": 10, "t1": 500e6, "t2": 500e6},
        {"x_est": 12, "t1": 750e6, "t2": 750e6},
        {"x_est": 14, "t1": 1e9,   "t2": 1e9}]

class TestLs(unittest.TestCase):

    def run_ls(self, impl):
        N    = 4
        T_ns = 250e3
        ls   = Ls(N, data)
        ls.process(impl=impl)

        # Results
        x_key    = "x_ls_" + impl
        y_key    = "y_ls_" + impl
        x_ls     = [r[x_key] for r in data if x_key in r]
        y_ls_ppb = [1e9*r[y_key] for r in data if y_key in r]

        np.testing.assert_almost_equal(x_ls, [12, 14])
        np.testing.assert_almost_equal(y_ls_ppb, [8, 8])

    def test_ls_t1(self):
        self.run_ls(impl="t1")

    def test_ls_t2(self):
        self.run_ls(impl="t2")

    def test_ls_eff(self):
        self.run_ls(impl="eff")

