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

    def process(self, impl="t2"):
        """Process the observations

        Using the raw time offset offset measurements and the Sync arrival
        timestamps, estimate the time and frequency offset of windows of
        samples.

        There are three distinct implementations for least-squares: "t2", "t1"
        and "eff", which are described next:

        - "t2" (default) : Uses timestamp "t2" when forming the observation
          matrix H.
        - "t1"           : Uses timestamp "t1" when forming the observation
          matrix H.
        - "eff"          : Computational-efficient implementation

        NOTE: The ideal time samples to be used in this matrix would be the true
        values of timestamps "t2", according to the reference time, not the
        slave time. Hence, the problem with using timestamps "t2" directly is
        that they are subject to slave impairments. When observing a long
        window, the last timestamp "t2" in the window may have drifted
        substantially with respect to the true "t2". In contrast, timestamps
        "t1" are taken at the master side, so they are directly from the
        reference time. However, the disadvantage of using "t1" is that they do
        not reflect the actual Sync arrival time due to PDV. Finally, the
        "efficient" choice ignores PDV and any timescale innacuracies to favour
        implementation simplicity.

        Args:
            t_choice : Timestamp choice when assemblign obervation matrix

        """

        n_data = len(self.data)
        N      = self.N

        # Vector of noisy time offset observations
        x_obs   = [res["x_est"] for res in self.data]

        # Vector of master timestamps
        t1 = [res["t1"] for res in self.data]

        # For "t1" and "t2", initialize vector of timestamps. For "eff",
        # initialize the matrix that is used for LS computations
        if (impl == "t1"):
            t = t1
        elif (impl == "t2"):
            t = [res["t2"] for res in self.data]
        elif (impl == "eff"):
            P = (2 / (N*(N+1))) * np.array([[(2*N - 1), -3], [-3, 6/(N-1)]]);
        else:
            raise ValueError("Unsupported LS timestamp mode")


        # Iterate over sliding windows of observations
        for i in range(0, n_data - N):
            # Window start and end indexes
            i_s = i
            i_e = i + N

            # Observation window
            x_obs_w = x_obs[i_s:i_e]

            # LS estimation
            if (impl == "eff"):
                # Accumulator 1
                if (i == 0):
                    Q_1   = np.sum(x_obs_w)
                else:
                    # Slide accumulator - throw away oldest and add new
                    Q_1   -= x_obs[i_s - 1]
                    Q_1   += x_obs[i_e]

                # Accumulator 2
                Q_2   = np.sum(np.multiply(np.arange(N), x_obs_w))
                # NOTE: we can't slide Q_2 like Q_1. This is because all weights
                # change from one window to the other.

                Q     = np.array([Q_1, Q_2])
                # LS Estimation
                Theta     = np.dot(P,Q.T);
                x0        = Theta[0]
                y_times_T = Theta[1]
                x_f       = x0 + (y_times_T * N)
            else:
                # Timestamps over observation window
                t_w     = t[i_s:i_e]
                tau = np.asarray([float(tt - t_w[0]) for tt in t_w])
                # Observation matrix
                H   = np.hstack((np.ones((N, 1)), tau.reshape(N, 1)))
                # NOTE: the observation matrix has to be assembled every time
                # for this approach. The "efficient" approach does not need to
                # re-compute H (doesn't even use H)

                # LS estimation
                x0, y = np.linalg.lstsq(H, x_obs_w, rcond=None)[0]
                # LS-fitted final time offset within window
                T_obs = float(t_w[-1] - t_w[0])
                x_f   = x0 + y * T_obs

            # Include LS estimations within the simulation data
            self.data[i_e - 1]["x_ls_" + impl] = x_f
            if (impl != "eff"):
                self.data[i_e - 1]["y_ls_" + impl] = y

