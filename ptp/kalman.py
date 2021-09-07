"""Kalman Filter
"""
import logging

import numpy as np

import ptp.cache

logger = logging.getLogger(__name__)


class KalmanFilter():
    def __init__(self,
                 data,
                 T,
                 N=1,
                 obs_model='scalar',
                 s_0=None,
                 P_0=None,
                 R=None,
                 Q=None):
        """Kalman Filter for Time/Frequency Offset Processing

        ---- Predict Step ----

        The Kalman state vector is:

        s[n] = [ x[n] ]
               [ y[n] ]

        that is, composed by the time offset (x[n]) and frequency offset (y[n]).

        The recursive model for the time offset is:

        x[n] = x[n-1] + y[n-1]*T + w_x[n],

        where w_x[n] is the time offset state noise.

        The recursive model for the frequency offset is:

        y[n] = y[n-1] + w_y[n],

        where w_y[n] is the frequency offset state noise.

        Thus, the state transition matrix becomes:

        A = [ 1   T ],
            [ 0   1 ]

        such that:

        s[n+1] = A*s[n] + w[n],

        where w[n] ~ N(0, Q) is the state noise vector and Q is the so-called
        state noise covariance matrix.

        The state noise covariance matrix takes the expected variability of the
        true state into account. The true frequency offset changes over time due
        to several oscillator noise sources. Similarly, the time offset
        uncertainty comes from phase noise and other effects. Both of these
        uncertainties are small in magnitude, so that the transition covariance
        matrix is expected to have small values.

        The model in [1] considers random-walk in both time and frequency. These
        are given in terms of normalized variances, which must be scaled by the
        observation period (Sync message period). Here, in contrast, we consider
        non-normalized variances, "var_y" and "var_x".

        Q = [ var_x,   0,  ]
            [   0,   var_y ]

        ---- Update Step ----

        In addition to the state model, the Kalman filter also uses a
        measurement model.

        The measurement model can rely on scalar observations, as presented in
        [2], where only the time offset is observed, or vector observations, as
        presented in [1], where both time and frequency offsets are observed.

        In the scalar-observation model, the observation is given by:

        z[n] = x[n] + v_x[n],

        where z[n], x[n] (time offset), and v_x[n] are all scalars.

        In the vector-observation model, it is given by:

        z[n] = [ x[n] ] + [ v_x[n] ]
               [ y[n] ]   [ v_y[n] ],

        where z[n] is (2x1).

        The observed values come directly from the raw time and frequency offset
        measurements that are taken after each delay request-response
        exchange. That is, z[n] = [x_est[n], y_est[n]]^T.

        The observation covariance matrix reflects the confidence on the
        observations. Both time and frequency offset observations are expected
        to be very noisy. Thus, it is important to define sufficiently high
        variances/covariances in this covariance matrix.

        When ignoring the timestamping noise (e.g., due to the timestamp
        granularity) and assuming that the PDV is the predominant noise, the
        observation noise covariance matrix can be shown to be given by:

        R = [ (Var{d_ms} + Var{d_sm})/4 ]

        when the scalar-observation model is used, or:

        R = [ (Var{d_ms} + Var{d_sm})/4 , Var{d_ms} / (2*N*T)    ]
            [  Var{d_ms} / (2*N*T)      , 2*Var{d_ms} / (N*T)**2 ]

        when the vector-observation model is used.

        The former (scalar-observation) is easier to compute in practice because
        it can be computed by "Var{d_est}", namely the variance of the delay
        estimates taken from PTP two-way exchanges. In contrast, the latter
        (vector-observation) requires the knowledge of the individual m-to-s and
        s-to-m variances, which a practical slave does not know. This
        formulation also assumes that m-to-s and s-to-m delays are independent
        (distinct) WSS and white discrete-time random processes.

        References:

        [1] G. Giorgi and C. Narduzzi, "Performance Analysis of
            Kalman-Filter-Based Clock Synchronization in IEEE 1588 Networks," in
            IEEE Transactions on Instrumentation and Measurement, vol. 60,
            no. 8, pp. 2902-2909, Aug. 2011.
        [2] G. Giorgi, "An Event-Based Kalman Filter for Clock Synchronization,"
            in IEEE Transactions on Instrumentation and Measurement, vol. 64,
            no. 2, pp. 449-457, Feb. 2015.
        [3] Brown, R. and Hwang, P., Introduction To Random Signals And Applied
            Kalman Filtering With Matlab Exercises, 4Th Edition. John Wiley &
            Sons, 2012.

        Args:
            data : Array of objects with simulation data
            T    : Nominal Sync period in sec
            N    : Interval used for frequency offset estimations
            s_0  : Initial state
            P_0  : Initial state's covariance matrix
            Q    : Process noise covariance matrix
            R    : Measurement covariance matrix

        """
        self.data = data  # this pointer could be changed
        self._original_data = data  # this pointer should be immutable
        self.T = T
        self.N = N
        self.obs_model = obs_model

        assert(obs_model in ['scalar', 'vector']), \
            f"Unknow observation model {obs_model}"

        # System measurements (scalar or vector)
        if (self.obs_model == 'scalar'):
            self._set_scalar_observation()
        elif (self.obs_model == 'vector'):
            self._set_vector_observation()

        # Default system state
        self.s_0 = self._set_initial_state() if s_0 is None else s_0
        self.P_0 = np.diag([self.R[0, 0], 1e10]) if P_0 is None else P_0
        # Note: In order to improve the initial Kalman transitory, initialize
        # the covariance matrix with the time offset variance R and a big
        # arbitrary number for the frequency offset variance. The objective is
        # to initialize the filter with a high gain. In other words, starts the
        # filter with low confidence in the state, favoring the observed
        # measurements.

        self.A = np.array([
            [1., self.T],  # State transition matrix
            [0., 1.]
        ])
        self.Q = np.array([
            [1e-13, 0],  # State noise cov matrix
            [0, 1e-18]
        ])

        # Validate args
        assert (s_0 is None or isinstance(s_0, np.ndarray))
        assert (P_0 is None or isinstance(P_0, np.ndarray))
        assert (Q is None or isinstance(Q, np.ndarray))
        assert (R is None or isinstance(R, np.ndarray))

        # Override some parameters based on optional constructor arguments
        self.Q = self.Q if Q is None else Q  # State noise cov matrix
        self.R = self.R if R is None else R  # Obs noise cov matrix

        # Define matrix dimensions
        self.dim_state = self.s_0.shape[0]
        self.dim_obs = 1 if self.obs_model == 'scalar' else self.z.shape[1]

        # Gain and residual error
        self.K = np.zeros((self.dim_state, self.dim_obs))  # Kalman gain
        self.y = np.zeros((self.dim_obs, 1))  # Residual error
        self.S = np.zeros((self.dim_obs, self.dim_obs))  # System uncertainty

        # Identity matrix
        self.I = np.eye(self.dim_state)

    def _set_initial_state(self):
        """Define the initial state vector

        Normally the initial state vector is random. However, here we choose to
        use the first measurements as the initial state. The goal is to reduce
        the convergence time.

        """
        # Estimate frequency offset based on the first two exchanges
        t1 = np.array([float(r["t1"]) for r in self.data[:2]])
        t2 = np.array([float(r["t2"]) for r in self.data[:2]])
        delta_master = float(np.diff(t1))
        delta_slave = float(np.diff(t2))
        y_est = (delta_slave - delta_master) / delta_master

        # First time offset raw estimate
        x_est = self.data[0]["x_est"]

        return np.array([x_est, 1e9 * y_est])

    def _reset_state(self):
        """Set the initial Kalman filter state

        The Kalman filter depends on the initial state vector s[-1] and its
        covariance matrix. If everything else is static (including
        measurements), as it is in this implementation, the filtering results
        derive from this initial state. Hence, here, reset the state vector and
        the state estimate covariance matrix.

        """
        self.s_post = self.s_0  # Initial a posteriori state vector
        self.P = self.P_0  # State estimate covariance matrix

    def _set_scalar_observation(self):
        """Set scalar observation model

        The scalar-observation model processes the time offset measurements.

        """
        # Vector containing all scalar measurements (time offset measurements)
        # to be processed throughout the filtering:
        self.z = np.array([r["x_est"] for r in self.data])
        assert (len(self.z) > 0)

        # Observation matrix: maps the state variables into the measurements
        #
        # In the scalar-observation model, the measurement is only composed by
        # the time offset, so the measurement transition matrix becomes the row
        # vector [1., 0], which neglects the frequency offset state and maps
        # only the time offset state to the time offset measurement.
        self.H = np.array([[1., 0.]])

        # Observation noise covariance matrix
        #
        # In the scalar observation model, the measurements noise covariance
        # matrix reduces to a scalar value, which corresponds to the variance of
        # the estimated delay experienced by the PTP messages. Interestingly,
        # this variance can be computed based on two-way delay measurements that
        # the PTP slave has in practice, as follows:
        d_est = np.array([r['d_est'] for r in self.data])
        var_d = np.var(d_est)
        self.R = np.array([[var_d]])

        # Check dimensions
        assert (self.H.shape == (1, 2))
        assert (self.R.shape == (1, 1))

    def _set_vector_observation(self):
        """Set vector observation model

        The vector-observation model processes both time and frequency offsets.

        """
        # Matrix containing all vector measurements to be processed throughout
        # the filtering. Each row is one vector measurement z[n] = [x_est[n],
        # y_est[n]], i.e., containing time and frequency offsets:
        x_ns = np.array([r["x_est"] for r in self.data if "y_est" in r])
        y = np.array([r["y_est"] * 1e9 for r in self.data if "y_est" in r])
        self.z = np.vstack((x_ns, y)).T
        assert (len(self.z) > 0)
        # NOTE: in the recursive time offset model (repeated below), x[n] is in
        # ns, while T is in seconds. Hence, y[n] must be in ppb (ns/sec), so
        # that y[n]*T yields ns.

        # Find where the frequency estimation starts
        for i, r in enumerate(self.data):
            if ("y_est" in r):
                i_obs_start = i
                break

        self.data = self._original_data[i_obs_start:]
        assert (all([("y_est" in r) for r in self.data]))

        # Observation matrix
        self.H = np.eye(2)

        # Observation noise covariance matrix
        #
        # FIXME: Use the estimated delay estimation here instead
        d_ms = np.array([r['d'] for r in self.data])
        d_sm = np.array([r['d_bw'] for r in self.data])
        var_x = (np.var(d_ms) + np.var(d_sm)) / 4
        var_y = (2 * np.var(d_ms)) / ((self.N * self.T)**2)
        cov_x_y = np.var(d_ms) / (2 * self.N * self.T)
        cov_y_x = cov_x_y
        self.R = np.array([[var_x, cov_x_y], [cov_y_x, var_y]])

        # Check dimensions
        assert (self.H.shape == (2, 2))
        assert (self.R.shape == (2, 2))

    def _eval_error(self,
                    q_vec,
                    error_metric,
                    early_stopping,
                    n_samples,
                    patience=15):
        """Evaluate error for a given process noise covariance matrix

        Args:
            q_vec          : Vector with covariance matrix to evaluate
            error_metric   : Chosen error metric: 'max-te' or 'mse'
            early_stopping : Whether to stop search when min{erro} stalls
            n_samples      : Number of samples to consider for the analysis,
                             so that the acceptable transient phase is neglected
                             on the error assessment.
            patience       : Number of consecutive iterations without
                             improvement to wait before signaling an early stop

        Returns:
            Q_best : Best evaluated covariance matrix
            error  : Vector with the error computed for all given variances

        """
        # Control variables
        last_print = 0
        min_err = np.inf
        patience_count = 0
        error = np.zeros(len(q_vec)**2)
        Q_matrix       = np.array([np.diag([var_x, var_y]) for var_x in q_vec \
                                   for var_y in q_vec])

        for i, Q in enumerate(Q_matrix):

            # Track progress
            progress = (i / len(Q_matrix))
            if (progress - last_print > 0.1):
                logger.info("Optimization progress {:5.2f} %".format(progress *
                                                                     100))
                last_print = progress

            # Run Kalman
            self.Q = Q
            self.process()

            # Get time offset estimation errors
            #
            # NOTE: use only the last n_samples, given that this is the subset
            # of samples that is targeted for analysis (after the acceptable
            # transient).
            x_err = np.array([
                r["x_kf"] - r["x"] for r in self.data if "x_kf" in r
            ])[-n_samples:]

            # Compute error based on different metrics. Note that the entire
            # "x_err" time series are used to compute the final error.
            if (error_metric == "max-te"):
                error[i] = np.amax(np.abs(x_err))
            elif (error_metric == "mse"):
                error[i] = np.square(x_err).mean()
            else:
                raise ValueError("Error metric %s is invalid" % (error_metric))

            # Keep track of the minimum error with "early stopping"
            #
            # Stop search if the Q matrix with minimum error remains the same
            # for a number of consecutive matrices.
            #
            # NOTE: patience count tracks the number of iterations with no error
            # reduction (or improvement).

            # Update min{error}
            if (error[i] < min_err):
                min_err = error[i]  # min error so far
                Q_best = Q  # best Q matrix so far
                patience_count = 0
            else:
                patience_count += 1

            if (early_stopping and patience_count > patience):
                break

        return Q_best, error

    def process(self, save_aux=False):
        """Process the observations"""
        logger.info("Processing Kalman Filter")

        # Remove previous estimates in case they already exist
        for r in self.data:
            r.pop("x_kf", None)
            r.pop("y_kf", None)
            r.pop("kf", None)

        logger.debug("s[-1] = {} \n"
                     "P[-1] = {} \n"
                     "H     = {} \n"
                     "Q     = {} \n"
                     "R     = {} \n".format(self.s_0, self.P_0, self.H, self.Q,
                                            self.R))

        # Reset the initial state
        self._reset_state()

        # Iterate over observations
        for i, z in enumerate(self.z):
            ##### Prediction #####

            # Predict the next state of the system (a priori)

            # A priori prediction of the state at the next time step
            self.s_prior = np.dot(self.A, self.s_post)

            # A priori state estimate's covariance matrix
            self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

            ##### Update #####

            # Update the next state estimate based on a new measurement

            # Residual error between the new measurement and the a priori
            # prediction
            self.y = z - np.dot(self.H, self.s_prior)

            # Kalman gain
            PHT = np.dot(self.P, self.H.T)
            self.S = np.dot(self.H, PHT) + self.R
            SI = np.linalg.pinv(self.S)
            self.K = np.dot(PHT, SI)

            # Update the state prediction (x) considering the new measurement:
            self.s_post = self.s_prior + np.dot(self.K, self.y)

            # Update the state prediction's covariance matrix
            #
            # The formulation usually seen in the literature is a "short form":
            #
            # P = (I - KH)P
            #
            # However, here we implement a "long form", also named as Joseph
            # update equation, that as showed in [3] is more numerically stable
            # and works for non-optimal K:
            #
            # P = (I-KH)P(I-KH)' + KRK'
            #
            I_KH = self.I - np.dot(self.K, self.H)
            KRK = np.dot(np.dot(self.K, self.R), self.K.T)
            self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + KRK

            # Save time and frequency offset estimates on global data records
            self.data[i]['x_kf'] = self.s_post[0]
            self.data[i]['y_kf'] = self.s_post[1] * 1e-9

            # Additional information useful for analysis
            if (save_aux):
                self.data[i]['kf'] = {}
                self.data[i]['x_kf_prior'] = self.s_prior[0]
                self.data[i]['y_kf_prior'] = self.s_prior[1] * 1e-9
                self.data[i]['kf']['P'] = self.P
                self.data[i]['kf']['K'] = self.K
                self.data[i]['kf']['S'] = self.S

    def optimize(self,
                 cache=None,
                 error_metric='mse',
                 early_stopping=True,
                 force=False,
                 skip=0.2):
        """Optimize the process noise covariance matrix

        Tries the filtering with several state noise covariance matrices and
        sets the internal matrix (Q) to the best evaluated value.

        Args:
            cache          : Cache handler used to save the optimized
                             configuration in a json file
            error_metric   : Chosen error metric: 'max-te' or 'mse'
            early_stopping : Whether to stop the search when min{erro} stalls
            patience       : Number of consecutive iterations without
                             improvement to wait before signaling an early stop
            skip           : Fraction of the dataset to skip during the error
                             assessment. Should be set to an acceptable
                             transient for analysis.

        Returns:

        """
        if (cache is not None):
            assert (isinstance(cache, ptp.cache.Cache)), "Invalid cache object"
            cached_cfg = cache.load('kf')

            if (cached_cfg and not force):
                self.Q = np.array(cached_cfg['Q'])
                return

        var_state_vec = np.array(
            [1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1])

        # Number of samples to consider when evaluating the error of each KF
        # configuration
        n_samples = int((1 - skip) * len(self.data))

        Q_best, _ = self._eval_error(var_state_vec,
                                     error_metric=error_metric,
                                     early_stopping=early_stopping,
                                     n_samples=n_samples)

        logger.info("Optimal process noise covariance mtx: {}".format(Q_best))

        if (cache is not None):
            cache.save({'Q': Q_best.tolist()}, identifier='kf')

        self.Q = Q_best
