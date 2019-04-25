"""Least-squares Estimator
"""
import numpy as np


class Ls():
    def __init__(self, N, data):
        """Least-squares Time Offset Estimator

        Args:
            N    : observation window length
            data : Array of objects with simulation data

        """
        self.N    = N
        self.data = data

    def process(self):
        """Process the observations

        Using the raw time offset offset measurements and the Sync arrival
        timestamps, estimate the time and frequency offset of windows of
        samples.

        """

        # Vector of noisy time offset observations
        x_obs   = [res["x_est"] for res in self.data]
        # Vector of Sync arrival timestamps
        t2      = [res["t2"] for res in self.data]

        # Iterate over sliding windows of observations
        for i in range(0, len(x_obs) - self.N):
            # Window start and end indexes
            i_s = i
            i_e = i + self.N

            # Observation window
            x_obs_w = x_obs[i_s:i_e]
            t2_w    = t2[i_s:i_e]

            # Observation matrix
            tau = np.asarray([float(t - t2_w[0]) for t in t2_w])
            A   = np.hstack((np.ones((self.N, 1)), tau.reshape(self.N, 1)))

            # LS estimation
            x0, y = np.linalg.lstsq(A, x_obs_w, rcond=None)[0]

            # LS-fitted final time offset within window
            T   = float(t2_w[-1] - t2_w[0])
            x_f = x0 + y * T

            # Include LS estimations within the simulation data
            self.data[i_e - 1]["x_ls"] = x_f
            self.data[i_e - 1]["y_ls"] = y*1e9

