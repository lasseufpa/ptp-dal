"""Kalman Filter
"""
import logging
import numpy as np
from pykalman import KalmanFilter

class Kalman():
    def __init__(self, data, T,
                 trans_cov = [[1, 0], [0, 1e-2]],
                 obs_cov = [[1e6, 0], [0, 1e6]]):
        """Kalman Filter for Time/Frequency Offset

        The Kalman state vector is:

        s = [ x ]
            [ y ]

        that is, composed by the time and frequency offset.

        The recursive model for the time offset is:

        x[n+1] = x[n] + y[n]*T

        Thus, the state transition matrix becomes:

        A = [ 1   T ],
            [ 0   1 ]

        such that:

        s[n+1] = A*s

        Accordingly, both time and frequency offsets are observed. That is, the
        observation vector is:

        y = [ x_tilde ]
            [ y_tilde ]

        The observed values come directly from the raw time offset measurements
        that are taken after each delay request-response.

        The transition covariance matrix takes the expected variability of the
        true state into account. The true frequency offset changes over time due
        to several oscillator noise sources. Similarly, the time offset
        uncertainty comes from phase noise and other effects. Both of these
        uncertainties are small in magnitude, so that the transition covariance
        matrix is expected to have small values.

        Meanwhile, the observation covariance matrix reflects the confidence on
        the observations. Both time and frequency offset observations are
        expected to be very noise. Thus, it is important to define sufficiently
        high variances/covariances in this covariance matrix.

        Args:
            data      : Array of objects with simulation data
            T         : Nominal Sync period in sec
            trans_cov : Transition covariance matrix
            obs_cov   : Observation covariance matrix

        """
        self.data = data
        self.T    = T

        trans_matrix  = [[1, T], [0, 1]]
        obs_matrix    = np.eye(2)
        trans_offsets = np.zeros(2)
        obs_offset    = np.zeros(2)

        # initial state (start with very little confidence)
        state_mean_0  = np.zeros(2)
        state_cov_0   = 1e5 * np.eye(2)

        self.kf = KalmanFilter(
            transition_matrices      = trans_matrix,
            observation_matrices     = obs_matrix,
            transition_covariance    = trans_cov,
            observation_covariance   = obs_cov,
            transition_offsets       = trans_offsets,
            observation_offsets      = obs_offset,
            initial_state_mean       = state_mean_0,
            initial_state_covariance = state_cov_0
        )

    def process(self):
        """Process the observations

        """

        logger = logging.getLogger("KF")

        # Vector of noisy time offset observations
        idx_vec    = np.array([r["idx"] for r in self.data if "y_est" in r])
        x_obs_ns   = np.array([r["x_est"] for r in self.data if "y_est" in r])
        y_obs_ppb  = np.array([1e9*r["y_est"] for r in self.data if "y_est" in r])
        n_data     = len(x_obs_ns)

        # Iterate over the observations
        state_mean = self.kf.initial_state_mean
        state_cov  = self.kf.initial_state_covariance
        for i, idx in enumerate(idx_vec):
            # feed a new observation into the filter
            obs = [x_obs_ns[i], y_obs_ppb[i]]

            state_mean, state_cov = self.kf.filter_update(
                state_mean,
                state_cov,
                obs)

            # put filtered results in the list of runner results
            self.data[idx]["x_kf"] = state_mean[0]
            self.data[idx]["y_kf"] = state_mean[1]*1e-9

            logger.debug("New state\tx_f: %f ns y: %f ppb" %(
                state_mean[0], state_mean[1]))

