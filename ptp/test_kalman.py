import unittest
import numpy as np
import copy
from pykalman import KalmanFilter as PyKalman
from ptp.kalman import *

immutable_data = [
    {"x_est": 6 , "y_est": 2, "d_est": 750, "d": 700,  "d_bw": 800},
    {"x_est": 8,  "y_est": 2, "d_est": 800, "d": 1000, "d_bw": 600},
    {"x_est": 10, "y_est": 2, "d_est": 500, "d": 600,  "d_bw": 400},
    {"x_est": 12, "y_est": 2, "d_est": 900, "d": 1000, "d_bw": 800},
    {"x_est": 14, "y_est": 2, "d_est": 500, "d": 400,  "d_bw": 600}
]

class TestKalman(unittest.TestCase):

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

        for i,d in enumerate(data):
            state_mean, state_cov = pykf.filter_update(
                state_mean,
                state_cov,
                z[i])

            d["x_pykf"] = state_mean[0]
            d["y_pykf"] = state_mean[1]
            d["P_pykf"] = state_cov

    def test_kf_scalar_model(self):
        """Compare our scalar-observation model to pykalman"""

        data  = copy.deepcopy(immutable_data)

        # Delay estimations used to set the observation noise variance
        d     = np.array([r["d_est"] for r in data])
        var_d = np.var(d)

        # Kalman matrices
        T     = 1
        A     = np.array([[1, T], [0, 1]])
        H     = np.array([[1., 0.]])
        Q     = np.array([[0.001, 0.005],[0.005, 0.01]])
        R     = np.array([[var_d]])

        # Initial state
        s_0   = np.array([data[0]["x_est"], 1e9*data[0]["y_est"]])
        P_0   = 1e3 * np.eye(2)

        # Measurements
        z     = np.array([r["x_est"] for r in data])

        # Run PyKalman
        self.run_pykalman(data, z, s_0, P_0, A, H, Q, R)

        # Run our module
        kf = KalmanFilter(data, T, obs_model='scalar', skip_transitory=False,
                          s_0=s_0, P_0=P_0, R=R, Q=Q)
        kf.process()

        for d in data:
            self.assertAlmostEqual(d["x_pykf"], d["x_kf"], places=6)
            self.assertAlmostEqual(d["y_pykf"], d["y_kf"]*1e9, places=6)
            np.testing.assert_almost_equal(d["P_pykf"], d["kf"]["P"])

    def test_kf_vector_model(self):
        """Compare our vector-observation model to pykalman"""

        data  = copy.deepcopy(immutable_data)

        # Delay estimations used to set the observation noise variance
        N       = 1
        T       = 1
        d_ms    = np.array([r['d'] for r in data])
        d_sm    = np.array([r['d_bw'] for r in data])
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
        s_0   = np.array([data[0]["x_est"], 1e9*data[0]["y_est"]])
        P_0   = 1e3 * np.eye(2)

        # Measurements
        z     = np.vstack((np.array([r["x_est"] for r in data]),
                           np.array([r["y_est"]*1e9 for r in data]))).T

        # Run PyKalman
        self.run_pykalman(data, z, s_0, P_0, A, H, Q, R)

        # Run our module
        kf = KalmanFilter(data, T, obs_model='vector', skip_transitory=False,
                          s_0=s_0, P_0=P_0, R=R, Q=Q)
        kf.process()

        for d in data:
            self.assertAlmostEqual(d["x_pykf"], d["x_kf"], places=6)
            self.assertAlmostEqual(d["y_pykf"], d["y_kf"]*1e9, places=6)
            np.testing.assert_almost_equal(d["P_pykf"], d["kf"]["P"])

    def test_kf_predict(self):
        """Test a priori Kalman prediction step"""

        data  = copy.deepcopy(immutable_data)

        # The kalman filter performs the prediction by computing x[k+1] = Ax[k]
        s_0       = np.array([data[0]["x_est"], data[0]["y_est"]])
        kf        = KalmanFilter(data, 1., obs_model='scalar',
                                 skip_transitory=False, s_0=s_0)
        kf._reset_state()

        for i in range(1,len(data)):
            kf._predict()
            kf.s_post = kf.s_prior
            x_pred    = kf.s_prior[0]
            y_pred    = kf.s_prior[1]
            self.assertEqual(x_pred, data[i]["x_est"])
            self.assertEqual(y_pred, data[i]["y_est"])

    def test_kf_optimization(self):
        """Test optimization of the Kalman state noise covariance matrix (Q)"""
        data  = copy.deepcopy(immutable_data)

        for r in data:
            x_est_noise = (r["d"] - r["d_bw"])/2
            r['x']      = r['x_est'] - x_est_noise

        # Parameters
        T = 1

        # Initial state
        s_0   = np.array([data[0]["x_est"], 1e9*data[0]["y_est"]])

        # Filtering using the default Q matrix
        kf = KalmanFilter(data, T, obs_model='scalar', skip_transitory=False,
                          s_0=s_0)
        kf.process()

        # MSE
        x_err_1 = np.array([r["x_kf"] - r["x"] for r in data if "x_kf" in r])
        mse_1   = np.square(x_err_1).mean()

        # Save the default (starting) noise cov matrix Q
        starting_Q = kf.Q

        # Find the optimal Q matrix
        kf.optimize(error_metric='mse')

        # Check that the optimal Q is not the default
        optimal_Q = kf.Q
        self.assertTrue((optimal_Q != starting_Q).any())

        # Evaluate the performance with the optimal Q
        kf.process()

        # MSE
        x_err_2 = np.array([r["x_kf"] - r["x"] for r in data if "x_kf" in r])
        mse_2   = np.square(x_err_2).mean()

        print(mse_1, mse_2)
        self.assertTrue(mse_2 < mse_1)

