import unittest
import numpy as np
import copy
from pykalman import KalmanFilter as PyKalman
from ptp.kalman import *

immutable_data = [
    {"d": 700,  "d_bw": 800},
    {"d": 1000, "d_bw": 600},
    {"d": 600,  "d_bw": 400},
    {"d": 1000, "d_bw": 800},
    {"d": 400,  "d_bw": 900},
    {"d": 800,  "d_bw": 1000},
    {"d": 600,  "d_bw": 700}
]

class TestKalman(unittest.TestCase):
    def setUp(self):
        """Create a dataset with timestamps in nanoseconds

        Take the delays that are given in the above data array and fill in the
        corresponding timestamps and time offset. Assume that the time offset
        changes linearly by a fixed amount per PTP exchange.

        Note: This function preserves the original global `immutable_data` list
        and returns a new array with the timestamp data filled in.

        """

        data  = copy.deepcopy(immutable_data)
        ds    = list()
        T     = 1
        y     = 2           # Constante frequency offset
        x     = 10          # Initialize the time offset
        d_t   = 250e6       # Interval between PTP exchanges
        d_t23 = 100         # Interval from t2 to t3 on the slave
        t1    = 0
        t2    = 0
        t3    = 0
        t4    = 0

        for idx, elem in enumerate(data):
            t1         += d_t
            t2          = t1 + x + elem['d']
            t3          = t2 + d_t23
            t4          = t3 - x + elem['d_bw']
            d_est       = ((t4 - t1) - (t3 - t2))/2
            x_est_noise = (elem["d"] - elem["d_bw"])/2

            elem['idx']   = idx
            elem['t1']    = t1
            elem['t2']    = t2
            elem['t3']    = t3
            elem['t4']    = t4
            elem['d_est'] = d_est
            elem['x']     = x
            elem['y']     = y
            elem['x_est'] = x + x_est_noise

            if (idx > 0):
                delta_master  = t1 - ds[idx - 1]['t1']
                delta_slave   = t2 - ds[idx - 1]['t2']
                elem['y_est'] = ((delta_slave - delta_master) / delta_master)

            ds.append(elem)
            x += y*T

        self.data = ds

    def run_pykalman(self, data, z, s_0, P, A, H, Q, R):
        """Run the Kalman filtering implementation from pykalman"""

        pykf = PyKalman(
            transition_matrices      = A,
            observation_matrices     = H,
            transition_covariance    = Q,
            observation_covariance   = R,
            initial_state_mean       = s_0,
            initial_state_covariance = P
        )

        state_mean = pykf.initial_state_mean
        state_cov  = pykf.initial_state_covariance

        for i, obs in enumerate(z):
            state_mean, state_cov = pykf.filter_update(
                state_mean,
                state_cov,
                obs)

            data[i]["x_pykf"] = state_mean[0]
            data[i]["y_pykf"] = state_mean[1]
            data[i]["P_pykf"] = state_cov

    def test_kf_state_initialization(self):
        """Check the definition of the initial state"""

        kf = KalmanFilter(self.data, T=1, obs_model='scalar')

        self.assertEqual(kf.s_0[0], self.data[0]["x_est"])
        self.assertEqual(kf.s_0[1], self.data[1]["y_est"]*1e9)

    def test_kf_scalar_model(self):
        """Compare our scalar-observation model to pykalman"""

        # Delay estimations used to set the observation noise variance
        d     = np.array([r["d_est"] for r in self.data])
        var_d = np.var(d)

        # Kalman matrices
        T     = 1
        A     = np.array([[1, T], [0, 1]])
        H     = np.array([[1., 0.]])
        Q     = np.array([[0.001, 0.005],[0.005, 0.01]])
        R     = np.array([[var_d]])

        # Initial state
        s_0   = np.array([self.data[0]["x_est"], 1e9*self.data[1]["y_est"]])
        P_0   = 1e3 * np.eye(2)

        # Measurements
        z     = np.array([r["x_est"] for r in self.data])

        # Run PyKalman
        self.run_pykalman(self.data, z, s_0, P_0, A, H, Q, R)

        # Run our module
        kf = KalmanFilter(self.data, T, obs_model='scalar', s_0=s_0, P_0=P_0,
                          R=R, Q=Q)
        kf.process(save_aux=True)

        for data in self.data:
            self.assertAlmostEqual(data["x_pykf"], data["x_kf"], places=6)
            self.assertAlmostEqual(data["y_pykf"], data["y_kf"]*1e9, places=6)
            np.testing.assert_almost_equal(data["P_pykf"], data["kf"]["P"])

    def test_kf_vector_model(self):
        """Compare our vector-observation model to pykalman"""

        # Delay estimations used to set the observation noise variance
        N       = 1
        T       = 1
        d_ms    = np.array([r['d'] for r in self.data])
        d_sm    = np.array([r['d_bw'] for r in self.data])
        var_x   = (np.var(d_ms) + np.var(d_sm)) / 4
        var_y   = (2. * np.var(d_ms))/((N * T)**2)
        cov_x_y = np.var(d_ms) / (2 * N * T)
        cov_y_x = cov_x_y

        # Kalman matrices
        A     = np.array([[1, T], [0, 1]])
        H     = np.eye(2)
        Q     = np.array([[0.001, 0.005],[0.005, 0.01]])
        R     = np.array([[var_x, cov_x_y],
                           [cov_y_x, var_y]])

        # Initial state
        s_0   = np.array([self.data[0]["x_est"], self.data[1]["y_est"]*1e9])
        P_0   = 1e3 * np.eye(2)

        # Measurements
        x_est = np.array([r["x_est"] for r in self.data if "y_est" in r])
        y_est = np.array([r["y_est"]*1e9 for r in self.data if "y_est" in r])
        z     = np.vstack((x_est, y_est)).T
        assert(len(z) > 0)

        # Find where the frequency estimation starts
        for i, r in enumerate(self.data):
            if ('y_est' in r):
                i_obs_start = i
                break

        # Run PyKalman
        self.run_pykalman(self.data[i_obs_start:], z, s_0, P_0, A, H, Q, R)

        # Run our module
        kf = KalmanFilter(self.data, T, obs_model='vector', s_0=s_0, P_0=P_0,
                          R=R, Q=Q)
        kf.process(save_aux=True)

        for data in self.data[i_obs_start:]:
            self.assertAlmostEqual(data["x_pykf"], data["x_kf"], places=6)
            self.assertAlmostEqual(data["y_pykf"], data["y_kf"]*1e9, places=6)
            np.testing.assert_almost_equal(data["P_pykf"], data["kf"]["P"])

    def test_kf_optimization(self):
        """Test optimization of the Kalman state noise covariance matrix (Q)"""

        # Filtering using the default Q matrix
        kf = KalmanFilter(self.data, T=1, obs_model='scalar')
        kf.process()

        # MSE
        x_err_def = np.array([r["x_kf"] - r["x"] for r in self.data if "x_kf" in r])
        mse_def   = np.square(x_err_def).mean()

        # Save the default (starting) noise cov matrix Q
        default_Q = kf.Q

        # Find the optimal Q matrix
        kf.optimize(error_metric='mse', early_stopping=False)
        kf.process()
        optimal_Q = kf.Q

        # MSE
        x_err_opt = np.array([r["x_kf"] - r["x"] for r in self.data if "x_kf" in r])
        mse_opt   = np.square(x_err_opt).mean()

        # Check that the optimal Q is not the default
        self.assertTrue((optimal_Q != default_Q).any())

        # Confirm that the optimization procedure improved the MSE
        self.assertTrue(mse_opt <= mse_def)

    def test_kf_optimization_early_stop(self):
        """Test early stopping option from the optimization procedure"""

        kf = KalmanFilter(self.data, T=1, obs_model='scalar')

        # Find the optimal Q matrix with early stopping active
        kf.optimize(error_metric='mse', early_stopping=True)
        kf.process()
        opt_Q_early_stop = kf.Q

        # Find the optimal Q matrix without early stopping
        kf.optimize(error_metric='mse', early_stopping=False)
        kf.process()
        opt_Q = kf.Q

        # Confirm that the early stopping mechanism successfully stops at the
        # optimal configuration.
        self.assertTrue((opt_Q == opt_Q_early_stop).all())

