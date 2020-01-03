"""Least-squares Estimator
"""
import logging
import numpy as np
logger = logging.getLogger(__name__)


class Ls():
    def __init__(self, N, data, T_ns=float('inf')):
        """Least-squares Time Offset Estimator

        Args:
            N    : observation window length (number of measurements per window)
            data : Array of objects with simulation data
            T_ns : nominal time offset measurement period in nanoseconds, used
                   for **debugging only**. It is used to obtain the fractional
                   frequency offset y (drift in sec/sec) when using the
                   efficient LS implementation, since the latter only estimates
                   y*T_ns (drif in nanoseconds/measurement). In the end, this is
                   used for plotting the frequency offset.

        """
        self.N       = N
        self.data    = data
        self.i_batch = 0

        # Learn the measurement period in case it is not provided
        self.T_ns = T_ns
        if (np.isinf(self.T_ns)):
            t1        = np.array([res["t1"] for res in data])
            self.T_ns = float(np.mean(np.diff(t1)))
            logger.warning("Automatically setting T_ns to %f ns", self.T_ns)

        # Matrix used by the efficient implementation
        self.P = (2 / (N*(N+1))) * np.array([[(2*N - 1), -3], [-3, 6/(N-1)]]);

    def _compute(self, x_obs, t=None, impl="eff", shift=1):
        """Compute least squares based on the observation vector

        Args:
            x_obs : Vector of time offset observations
            t     : Vector of Sync departure or arrival timestamps. Departure
                    when impl=="t1" and arrival when impl=="t2". It can be empty
                    (None) when impl=="eff".
            impl  : Least-squares implementation ("eff", "t2" or "t1")
            shift : Determines the shift (in samples) between consecutive
                    windows or, equivalently, the window overlap. A shift 1
                    means that a window of N samples contains N-1 samples from
                    the previous window (i.e. is fully overlapping).

        NOTE: The ideal Sync timestamps to be used in the observation matrix H
        (see implementation below) would be the true values of timestamps "t2",
        according to the reference time, not the slave time. However, the
        problem with using timestamps "t2" directly is that they are subject to
        slave impairments. When observing a long window, the last timestamp "t2"
        in the window may have drifted substantially with respect to the true
        "t2". In contrast, timestamps "t1" are taken at the master side, so they
        are directly from the reference time. However, the disadvantage of using
        "t1" is that they do not reflect the actual Sync arrival time after the
        message's PDV. The "efficient" choice ignores PDV and any timescale
        innacuracies in order to favour implementation simplicity.

        Args:
            t_choice : Timestamp choice when assembling obervation matrix

        """

        if (impl != "eff"):
            assert(t is not None)

        n_data      = len(x_obs)
        N           = self.N
        win_overlap = N - shift  # Window overlap
        new_per_win = shift      # new samples per window
        n_windows   = int((n_data - win_overlap)/new_per_win)

        logger.debug("Compute batch %d with %d windows" %(
            self.i_batch, n_windows))
        self.i_batch += 1

        # Vectorized and efficient implementation
        #
        # NOTE: the non-vectorized but efficient implementation can still be
        # found below, as it helps for understanding. However, the following
        # vectorized implementation is the one that is effectively used, as it
        # is much faster.
        if (impl == "eff"):
            # Stack overlapping windows into columns of a matrix
            X_obs     = np.zeros(shape=(N, n_windows))
            for i in range(N):
                X_obs[i, :] = x_obs[i:(i+n_windows):new_per_win]

            # Q1 and Q2 accumulator values for each window
            Q      = np.zeros(shape=(2, n_windows))
            Q[0,:] = np.sum(X_obs, axis=0)
            Q[1,:] = np.dot(np.arange(N), X_obs)

            # LS estimations
            Theta        = np.dot(self.P,Q)
            X0           = Theta[0,:]
            Y_times_T_ns = Theta[1,:]
            Xf           = X0 + (Y_times_T_ns * (N-1))
            Y            = Y_times_T_ns / self.T_ns

            return Xf, Y

        # Preallocate results (one for each window)
        Xf = np.zeros(n_windows)
        Y  = np.zeros(n_windows)

        # Iterate over sliding windows of observations
        for i in range(0, n_windows):
            # Window start and end indexes
            i_s = i * new_per_win
            i_e = i_s + N

            # Observation window
            x_obs_w = x_obs[i_s:i_e]

            # LS estimation
            if (impl == "eff"):
                assert(new_per_win == 1),\
                    "This LS implementation only works with overlapping windows"
                # Accumulator 1
                if (i == 0):
                    Q_1   = np.sum(x_obs_w)
                else:
                    # Slide accumulator - throw away oldest and add new
                    Q_1 -= x_obs[i_s - 1]
                    Q_1 += x_obs[i_e -1]
                # Accumulator 2
                if (i == 0):
                    Q_2 = np.sum(np.multiply(np.arange(N), x_obs_w))
                else:
                    # See derivation in Igor Freire's thesis, Section 3.6
                    Q_2 -= Q_1
                    Q_2 += N * x_obs[i_e -1]
                # Accumulator vector
                Q     = np.array([Q_1, Q_2])
                # LS Estimation
                Theta        = np.dot(P,Q.T);
                x0           = Theta[0] # initial time offset within window
                y_times_T_ns = Theta[1] # drift in nanoseconds/measurement
                # Fit the final time offset within the current window
                x_f          = x0 + (y_times_T_ns * (N-1))
                # Fractional frequency offset
                y            = y_times_T_ns / self.T_ns
            else:
                # Timestamps over observation window
                t_w = t[i_s:i_e]
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
            Xf[i] = x_f
            Y[i]  = y

        return Xf, Y

    def process(self, impl="eff", batch_mode=True, batch_size=4096):
        """Process the observations

        Using the raw time offset offset measurements and potentially also the
        Sync arrival/departure timestamps, estimate the time and frequency
        offset of windows of samples.

        Args:

            impl       : Least-squares implementation. There are three distinct
                         implementations for least-squares: "t2", "t1" and
                         "eff", which are described next:

                           "t2" (default) : Uses timestamp "t2" when forming the
                                            observation matrix H.
                           "t1"           : Uses timestamp "t1" when forming the
                                            observation matrix H.
                           "eff"          : Computationally-efficient
                                            implementation.

            batch_mode : Whether to process observation windows in batches,
                         rather than trying to process all windows at once.

            batch_size : Number of observation windows that compose a batch.

        """

        assert(impl in ["t1", "t2", "eff"]), "Unsupported LS timestamp mode"

        logger.info("Processing with N=%d" %(self.N))

        n_data      = len(self.data)
        win_overlap = self.N - 1 # samples repeated on a window from past window
        new_per_win = self.N - win_overlap # new samples per window
        # NOTE: assume each window of size N has N-1 entries from the previous
        # window (i.e. fully overlapping windows)

        # Corresponding number of windows and batches
        n_windows  = int((n_data - win_overlap)/new_per_win)
        n_batches  = np.ceil(n_windows/batch_size) if batch_mode else 1
        batch_size = batch_size if batch_mode else n_windows

        # Iterate over batches
        for i_w_s in range(0, n_windows, batch_size):
            i_w_e = i_w_s + batch_size # last window of this batch

            # Sample range covered by the windows
            i_s = i_w_s * new_per_win                       # 1st of the 1st window
            i_e = i_s + (batch_size-1)*new_per_win + self.N # last of the last

            # Vector of noisy time offset observations
            x_obs = np.array([res["x_est"] for res in self.data[i_s:i_e]])

            # For impl="t1" or impl="t2", vector of timestamps:
            if (impl == "t1"):
                t = np.array([res["t1"] for res in self.data[i_s:i_e]])
            elif (impl == "t2"):
                t =  np.array([res["t2"] for res in self.data[i_s:i_e]])
            elif (impl == "eff"):
                t = None

            # Compute LS for each window of this batch:
            Xf, Y = self._compute(x_obs, t, impl)

            assert(len(Xf) == batch_size or i_w_e > n_windows)
            assert(len(Xf) == len(Y))

            for i in range(0, len(Xf)):
                first_idx_in_window = i_s + i*new_per_win
                last_idx_in_window  = first_idx_in_window + self.N - 1
                self.data[last_idx_in_window]["x_ls_" + impl] = Xf[i]
                self.data[last_idx_in_window]["y_ls_" + impl] = Y[i]

