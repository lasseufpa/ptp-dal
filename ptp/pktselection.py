import logging
import numpy as np
from scipy import stats
logger = logging.getLogger(__name__)


SAMPLE_MODE_BIN_0 = 10 # starting value


class PktSelection():

    def __init__(self, N, data):
        """Packet Selection

        Args:
            N       : Observation window length
            data    : Array of objects with simulation data
        """

        self.N    = N
        self.data = data

        # Recursive moving-average
        self._movavg_accum    = 0              # recursive accumulator
        self._movavg_buffer   = np.zeros(2*N)  # circular buffer
        self._movavg_i        = N              # head index of the buffer

        # Exponentially-weight moving average
        self._ewma_alpha    = 1/N
        self._ewma_beta     = 1 - (1/N)
        self._ewma_last_avg = 0
        self._ewma_n        = 0 # sample index for bias correction

        # Sample-mode params
        self._sample_mode_bin = SAMPLE_MODE_BIN_0
        self._mode_stall_cnt  = 0

    def _window(self, v, N, shift = 1, copy = False):
        """Split numpy vector into overlapping windows

        From https://stackoverflow.com/a/45730836/2859410

        Args:
            v     : Numpy vector to split
            N     : Target window length
            shift : Controls the shift between consecutive windows or,
                    equivalently, the overlap. For instance, if shift=1, each
                    window overlaps with N-1 samples of the previous window.
            copy  : Set True to write to the array that is returned by this
                    function, as otherwise the returned array is a
                    memory-sharing view of the same numpy array (like the
                    result of a numpy.reshape).

        Returns:
            Matrix with overlapping windows. Each line will correspond to a
            window with N columns.

        """
        sh   = (v.size - N + 1, N)
        st   = v.strides * 2
        view = np.lib.stride_tricks.as_strided(v,
                                               strides = st,
                                               shape = sh)[0::shift]
        if copy:
            return view.copy()
        else:
            return view

    def _sample_avg_normal(self, x_obs):
        """Calculate the average of a given time offset vector

        Args:
            x_obs   : Vector of time offset observations

        Returns:
            The average of the time offset vector
        """

        return np.mean(x_obs)

    def _sample_avg_normal_vec(self, x_obs):
        """Calculate the sample-average of time offset observation windows

        Args:
            x_obs   : Matrix with time offset observations windows

        Returns:
            Vector with the average of each individual time offset observation
            window

        """

        return np.mean(x_obs, axis=1)

    def _sample_avg_recursive(self, x_obs):
        """Calculate the average of a given time offset vector recursively

        Uses a circular (ring) buffer with size 2*N, where the tail pointer is
        always lagging the head pointer by N. The head pointer is used to save
        new values and the tail pointer is used to throw away the oldest value.

        Args:
            x_obs   : Scalar time offset observation

        Returns:
            The moving average

        """

        i_head                      = self._movavg_i
        i_tail                      = (i_head - self.N) % (2*self.N)
        # Put new observation on the head of the buffer
        self._movavg_buffer[i_head] = x_obs
        self._movavg_accum         += x_obs
        # Remove the oldest value on the tail of the buffer
        self._movavg_accum         -= self._movavg_buffer[i_tail]
        # Compute new average and advance head pointer
        new_avg                     = self._movavg_accum / self.N
        self._movavg_i              = (self._movavg_i + 1) % (2*self.N)
        return new_avg

    def _sample_avg_ewma(self, x_obs):
        """Calculate the exponentially weighted moving average (EWMA)

        Args:
            x_obs   : Scalar time offset observation

        Returns:
            The bias-corrected exponentially weighted moving average
        """
        new_avg             = (self._ewma_beta * self._ewma_last_avg) + \
                              self._ewma_alpha * x_obs
        # Save for the next iteration
        self._ewma_last_avg = new_avg
        # Apply bias correction (but don't save the bias-corrected average)
        self._ewma_n       += 1
        bias_corr           = 1 / (1 - (self._ewma_beta ** self._ewma_n))
        corr_avg            = new_avg * bias_corr
        return corr_avg

    def _sample_median(self, t2_minus_t1_w, t4_minus_t3_w):
        """Calculate the sample-median estimate of a given observation window

        Args:
            t2_minus_t1_w : Vector of (t2 - t1) differences
            t4_minus_t3_w : Vector of (t4 - t3) differences

        Returns:
            The time offset estimate based on sample-median
        """

        t2_minus_t1 = np.median(t2_minus_t1_w)
        t4_minus_t3 = np.median(t4_minus_t3_w)
        x_est       = (t2_minus_t1 - t4_minus_t3)/2
        return x_est

    def _sample_median_vec(self, t2_minus_t1_mtx, t4_minus_t3_mtx):
        """Calculate the sample-median estimates of given observation windows

        Args:
            t2_minus_t1_mtx : Matrix with windows (as lines) containing
                              (t2 - t1) differences
            t4_minus_t3_mtx : Matrix with windows (as lines) containing
                              (t4 - t3) differences

        Returns:
            The individual time offset estimates of each observation window
            based on sample-median

        """

        t2_minus_t1 = np.median(t2_minus_t1_mtx, axis=1)
        t4_minus_t3 = np.median(t4_minus_t3_mtx, axis=1)
        x_est       = (t2_minus_t1 - t4_minus_t3)/2
        return x_est

    def _eapf(self, t2_minus_t1_w, t4_minus_t3_w):
        """Compute the time offset based on earliest arrivals

        Apply an earliest arrival packet filter (EAPF) on both master-to-slave
        Sync messages and slave-to-master Delay_Req messages to compute time
        offset.

        Within a window of N message exchanges, find the earliest-arriving Sync
        and the earliest-arriving Delay_Req, that is, the ones that were subject
        to minimum delay. To find the Sync, observe the index where the
        difference (t2 - t1) was the minimum. To find the Delay_Req, look for
        the minimum of the difference (t4 - t3). In the end, use these minimum
        differences to compute the time offset estimate.

        Note that not necessarily the index of minimum (t2 - t1) must be the
        same as the index of minimum (t4 - t3), as long as the time offset
        remains sufficiently constant within the window.

        Args:
            t2_minus_t1_w : Vector of (t2 - t1) differences
            t4_minus_t3_w : Vector of (t4 - t3) differences

        Returns:
           The time offset estimate

        """
        t2_minus_t1 = np.amin(t2_minus_t1_w)
        t4_minus_t3 = np.amin(t4_minus_t3_w)
        x_est       = (t2_minus_t1 - t4_minus_t3)/2

        return x_est

    def _eapf_vec(self, t2_minus_t1_mtx, t4_minus_t3_mtx):
        """Compute the time offset based on earliest arrivals

        Vectorized version of the above (`_eapf`). Instead of processing a
        single observation window, this function processes a matrix with many
        (or all) observation windows.

        Args:
            t2_minus_t1_mtx : Matrix of (t2 - t1) differences
            t4_minus_t3_mtx : Matrix of (t4 - t3) differences

        Returns:
           Vector with the time offset estimates corresponding to each
           observation window

        """
        t2_minus_t1 = np.amin(t2_minus_t1_mtx, axis=1)
        t4_minus_t3 = np.amin(t4_minus_t3_mtx, axis=1)
        x_est       = (t2_minus_t1 - t4_minus_t3)/2

        return x_est

    def _sample_maximum(self, t2_minus_t1_w, t4_minus_t3_w):
        """Compute the time offset based on latest arrivals

        Implement the same logic of EAPF, but with the difference that the
        maximum delay is pursued rather than the minimum.

        Args:
            t2_minus_t1_w : Vector of (t2 - t1) differences
            t4_minus_t3_w : Vector of (t4 - t3) differences

        Returns:
           The time offset estimate

        """
        t2_minus_t1 = np.amax(t2_minus_t1_w)
        t4_minus_t3 = np.amax(t4_minus_t3_w)
        x_est       = (t2_minus_t1 - t4_minus_t3)/2
        return x_est

    def _sample_maximum_vec(self, t2_minus_t1_mtx, t4_minus_t3_mtx):
        """Compute the time offset based on latest arrivals

        Vectorized version of the above (`_sample_maximum`). Instead of
        processing a single observation window, this function processes a matrix
        with many (or all) observation windows.

        Args:
            t2_minus_t1_mtx : Matrix of (t2 - t1) differences
            t4_minus_t3_mtx : Matrix of (t4 - t3) differences

        Returns:
           Vector with the time offset estimates corresponding to each
           observation window

        """
        t2_minus_t1 = np.amax(t2_minus_t1_mtx, axis=1)
        t4_minus_t3 = np.amax(t4_minus_t3_mtx, axis=1)
        x_est       = (t2_minus_t1 - t4_minus_t3)/2
        return x_est

    def _sample_mode(self, t2_minus_t1_w, t4_minus_t3_w, cnt_threshold=3,
                     stall_patience=10):
        """Compute the time offset based on sample-mode

        Regarding the bin adjustment algorithm, note that a higher
        `cnt_threshold` will lead to more frequent adjustments (enlargements) of
        the bin width. A lower patience will also enlarge the bin width more
        frequently. Moreover, note that if the bin becomes too wide, the
        estimation resolution reduces. Hence, loosely speaking the
        `cnt_threshold` should be relatively small and `stall_patience`
        moderately higher.

        Args:
            t2_minus_t1_w  : Vector of (t2 - t1) differences
            t4_minus_t3_w  : Vector of (t4 - t3) differences
            cnt_threshold  : Minimum number of ocurrences on the mode bin
            stall_patience : Number of consecutive iterations of stalled mode
                             bin ocurrence count before the bin width is
                             adjusted

        Returns:
           The time offset estimate

        """
        bin_width      = self._sample_mode_bin
        half_bin_width = 0.5 * bin_width

        # Quantize timestamp difference vectors
        t2_minus_t1_q = np.round(t2_minus_t1_w / bin_width)
        t4_minus_t3_q = np.round(t4_minus_t3_w / bin_width)

        # Find the mode for (t2 - t1)
        (_, idx, counts) = np.unique(t2_minus_t1_q, return_index=True,
                                     return_counts=True)
        mode_cnt_fw      = np.amax(counts)
        mode_idx_fw      = idx[np.argmax(counts)]
        t2_minus_t1      = t2_minus_t1_q[mode_idx_fw] * bin_width + \
                           half_bin_width

        # Find the mode for (t4 - t3)
        (_, idx, counts) = np.unique(t4_minus_t3_q,
                                     return_index=True, return_counts=True)
        mode_cnt_bw      = np.amax(counts)
        mode_idx_bw      = idx[np.argmax(counts)]
        t4_minus_t3      = t4_minus_t3_q[mode_idx_bw] * bin_width + \
                           half_bin_width

        # Detect when we can't find a mode
        #
        # In case the occurrence count of the mode bin is below a threshold for
        # some consecutive number of iterations (denominated patience), we infer
        # that the mode detection is stalled and increase the bin width.
        if (mode_cnt_fw < cnt_threshold and mode_cnt_bw < cnt_threshold):
            self._mode_stall_cnt += 1
        else:
            self._mode_stall_cnt = 0

        if (self._mode_stall_cnt > stall_patience):
            self._sample_mode_bin += 10
            self._mode_stall_cnt   = 0

        # Final estimate
        x_est = (t2_minus_t1 - t4_minus_t3)/2
        return x_est

    def _sample_mode_vec(self, t2_minus_t1_mtx, t4_minus_t3_mtx,
                         cnt_threshold=3):
        """Compute the time offset based on sample-mode

        Vectorized sample-mode implementation. Its result differs from the
        non-vectorized sample-mode implementation due to how the bin adjustment
        is implemented.

        Args:
            t2_minus_t1_mtx : Matrix of (t2 - t1) differences
            t4_minus_t3_mtx : Matrix of (t4 - t3) differences
            cnt_threshold   : Minimum number of ocurrences on the mode bin

        Returns:
           Vector with the time offset estimates corresponding to each
           observation window

        """

        # Bin widths for t2-t1 and t4-t3 are independent in this implementation
        bin_width_fw = SAMPLE_MODE_BIN_0
        bin_width_bw = SAMPLE_MODE_BIN_0

        # Tune the bin width based on the first 100 observation windows
        done = False
        while (not done):
            # Quantize timestamp difference matrices
            t2_minus_t1_q = np.round(t2_minus_t1_mtx[:100,:] / bin_width_fw)
            t4_minus_t3_q = np.round(t4_minus_t3_mtx[:100,:] / bin_width_bw)

            # Find the mode
            t2_minus_t1_mode, mode_cnt_fw = stats.mode(t2_minus_t1_q, axis=1)
            t4_minus_t3_mode, mode_cnt_bw = stats.mode(t4_minus_t3_q, axis=1)

            # Adjust the mode bin such that less than 10% of the realizations
            # have a maximum mode count below threshold.
            done = True
            if ((mode_cnt_fw < cnt_threshold).sum() > 10):
                bin_width_fw += 10
                done = False

            if ((mode_cnt_bw < cnt_threshold).sum() > 10):
                bin_width_bw += 10
                done = False

        logger.info("t2-t1 bin was adjusted to %d" %(bin_width_fw))
        logger.info("t4-t3 bin was adjusted to %d" %(bin_width_bw))
        half_bin_width_fw = 0.5 * bin_width_fw
        half_bin_width_bw = 0.5 * bin_width_bw

        # Quantize timestamp difference matrices
        t2_minus_t1_q = np.round(t2_minus_t1_mtx / bin_width_fw)
        t4_minus_t3_q = np.round(t4_minus_t3_mtx / bin_width_bw)

        # Find the mode
        t2_minus_t1_mode, mode_cnt_fw = stats.mode(t2_minus_t1_q, axis=1)
        t4_minus_t3_mode, mode_cnt_bw = stats.mode(t4_minus_t3_q, axis=1)

        # Descale to revert quantization
        t2_minus_t1 = t2_minus_t1_mode * bin_width_fw + half_bin_width_fw
        t4_minus_t3 = t4_minus_t3_mode * bin_width_bw + half_bin_width_bw

        # Final estimates
        x_est = (t2_minus_t1 - t4_minus_t3)/2

        return x_est.reshape(x_est.size)

    def set_window_len(self, N):
        """Change the window length

        Args:
            N : new window length N
        """
        self.N              = N
        # Reset internal variables (some depend on N)
        self._movavg_accum  = 0
        self._movavg_buffer = np.zeros(2*N)
        self._movavg_i      = N
        self._ewma_alpha    = 1/N
        self._ewma_beta     = 1 - (1/N)
        self._ewma_last_avg = 0
        self._ewma_n        = 0
        self._sample_mode_bin = SAMPLE_MODE_BIN_0
        self._mode_stall_cnt  = 0

    def process(self, strategy, avg_impl="recursive", drift_comp=True,
                vectorize=True):
        """Process the observations

        Using the raw time offset measurements, estimate the time offset using
        sample-average ("average"), EWMA ("ewma"), sample-median ("median") or
        sample-minimum ("min") over sliding windows of observations.

        If drift compensation is enabled, use vector of drifts to compensate the
        differences of timestamps along the observation window.

        Note that:
            x[n] = t2[n] - (t1[n] + dms[n])
            x[n] = (t3[n] + dsm[n]) - t4[n]

        so that

            t2[n] - t1[n] = +x[n] + dms[n],
            t4[n] - t3[n] = -x[n] + dsm[n]

        However, x[n] can be modeled as:

            x[n] = x_0 + drift[n],

        where the drift[n] term is mostly due to frequency offset, which
        sometimes can be accurately estimated (especially for high quality
        oscillators).

        Using this model, and if the drift is compensated, it can be stated
        that:

            t2[n] - t1[n] - drift[n] = +x_0 + dms[n],
            t4[n] - t3[n] + drift[n] = -x_0 + dsm[n]

        In the end, this means that drift-compensated timestamp differences will
        lead to an approximately constant time offset corrupted by variable
        delay realizations. This is the ideal input to packet selection
        operators. Once the operators process the drift-compensated timestamp
        differences, ideally its result would approach:

            x_est -> x_0

        To complete the work, the total (cumulative) drift over the observation
        window should be re-added to the estimate, so that it approaches the
        true time offset by the end of the observation window (while `x_0` is
        the offset in the beginning):

            x_est += drift[N-1]

        For the operators that process time offset observations x_est[n]
        directly, the drift can be similarly removed prior to the selection
        operator and the sum of the drift re-added to the final result.

        Args:
            strategy   : Select the strategy of interest.
            avg_impl   : Sample-avg implementation ("recursive" or "normal").
                         The recursive implementation theoretically produces the
                         same result as the "normal" implementation, except
                         during initialization, due to the transitory of the
                         recursive implementation. However, the recursive one
                         is significantly more CPU efficient.
            drift_comp : Whether to compensate drift of timestamp differences or
                         time offset measuremrents prior to computing packet
                         selection operators.
            vectorize  : Whether to use vectorized implementation of selection
                         operators. When enabled, all observation windows will
                         be stacked as lines of a matrix and the full matrix
                         will be processed at once. Otherwise, each observation
                         window is processed independently.

        """

        if (drift_comp):
            drift_msg = " and drift compensation"
        else:
            drift_msg = ""

        logger.info("Processing sample-%s with N=%d" %(strategy, self.N) +
                    drift_msg)

        # Select vector of noisy time offset observations and delay estimation
        x_obs = np.array([res["x_est"] for res in self.data])
        d_obs = np.array([res["d_est"] for res in self.data])
        n_data = len(x_obs)

        # Assume that all elements of self.data contain "x_est" and "d_est"
        assert(len(x_obs) == len(self.data))
        assert(len(d_obs) == len(self.data))

        # Vector of time offset incremental drifts due to freq. offset:
        if (drift_comp):
            drift_est = np.zeros(x_obs.shape)
            for i in range(0, n_data):
                if ("drift" in self.data[i]):
                    drift_est[i] = self.data[i]["drift"]

            assert(drift_est.any()), "Drift estimations not available"

        if (strategy == 'average' and avg_impl == 'recursive'):
            # Sample-by-sample processing
            if (drift_comp):
                # Accumulate drift continuously and try to average out
                # drift-removed time offset estimates
                drift_accum = 0
                for i in range(0, n_data):
                    drift_accum += drift_est[i]
                    x_est = self._sample_avg_recursive(x_obs[i] - drift_accum)
                    self.data[i]["x_pkts_average"] = x_est + drift_accum
            else:
                for i in range(0, n_data):
                    x_est = self._sample_avg_recursive(x_obs[i])
                    self.data[i]["x_pkts_average"] = x_est
        elif (strategy == 'ewma'):
            # Sample-by-sample processing
            if (drift_comp):
                # Accumulate drift continuously and try to average out
                # drift-removed time offset estimates
                drift_accum = 0
                for i in range(0, n_data):
                    drift_accum += drift_est[i]
                    x_est = self._sample_avg_ewma(x_obs[i] - drift_accum)
                    self.data[i]["x_pkts_ewma"] = x_est + drift_accum
            else:
                for i in range(0, n_data):
                    x_est = self._sample_avg_ewma(x_obs[i])
                    self.data[i]["x_pkts_ewma"] = x_est
        elif (vectorize):
            # Vectorized processing - all observation windows are stacked as
            # lines of a matrix

            # Operator that processes time offset measurement windows
            if (strategy == 'average' and avg_impl == 'normal'):
                x_obs_mtx = self._window(x_obs, self.N, copy=True)

                # Remove drift
                if (drift_comp):
                    drift_cum_mtx  = self._window(drift_est,
                                                  self.N).cumsum(axis=1)
                    x_obs_mtx     -= drift_cum_mtx
                    drift_offsets  = drift_cum_mtx[:,-1]

                # Sample-average operator
                x_est = self._sample_avg_normal_vec(x_obs_mtx)

            # Operators that process timestamp difference windows
            else:
                # Timestamp differences
                t2_minus_t1 = np.array([float(r["t2"] - r["t1"])
                                        for r in self.data])
                t4_minus_t3 = np.array([float(r["t4"] - r["t3"])
                                        for r in self.data])

                # Form observation windows with timestamp differences
                t2_minus_t1_mtx = self._window(t2_minus_t1, self.N, copy=True)
                t4_minus_t3_mtx = self._window(t4_minus_t3, self.N, copy=True)

                # Remove drift
                if (drift_comp):
                    drift_cum_mtx    = self._window(drift_est,
                                                    self.N).cumsum(axis=1)
                    t2_minus_t1_mtx -= drift_cum_mtx
                    t4_minus_t3_mtx += drift_cum_mtx
                    drift_offsets    = drift_cum_mtx[:,-1]

                # Apply Selection operator
                if (strategy == 'median'):
                    x_est = self._sample_median_vec(t2_minus_t1_mtx,
                                                    t4_minus_t3_mtx)
                elif (strategy == 'min'):
                    x_est = self._eapf_vec(t2_minus_t1_mtx,
                                           t4_minus_t3_mtx)
                elif (strategy == 'max'):
                    x_est = self._sample_maximum_vec(t2_minus_t1_mtx,
                                                     t4_minus_t3_mtx)
                elif (strategy == 'mode'):
                    x_est = self._sample_mode_vec(t2_minus_t1_mtx,
                                                  t4_minus_t3_mtx)
                else:
                    raise ValueError("Strategy choice %s unknown" %(strategy))

            assert(len(x_est) == (n_data - self.N) + 1)

            # Re-add "quasi-deterministic" drift to estimates
            if (drift_comp):
                x_est += drift_offsets

            # Save results on global data records
            for i in range(0, (n_data - self.N) + 1):
                self.data[i + self.N - 1]["x_pkts_{}".format(strategy)] = x_est[i]
        else:
            # Window-by-window processing
            for i in range(0, (n_data - self.N) + 1):
                # Window start and end indexes
                i_s = i
                i_e = i + self.N

                # Observation window
                x_obs_w = x_obs[i_s:i_e]
                d_obs_w = d_obs[i_s:i_e]

                # Drift vector is the cumulative time offset drift due to
                # frequency offset relative to the beginning of the observation
                # window. Say if first sample of the window has an incremental
                # drift of 5 ns and second sample has a drift of 6 ns, the first
                # two elements of the drift vector below will be 5 and 11 ns.
                if (drift_comp):
                    drift_w = drift_est[i_s:i_e].cumsum()

                # Operator that processes time offset measurement windows
                if (strategy == 'average' and avg_impl == 'normal'):
                    # Remove drift from observed time offset
                    if (drift_comp):
                        x_obs_w = x_obs_w - drift_w

                    x_est = self._sample_avg_normal(x_obs_w)

                # Operator that processes timestamp difference windows
                else:
                    t2_minus_t1_w = np.array([float(r["t2"] - r["t1"])
                                              for r in self.data[i_s:i_e]])
                    t4_minus_t3_w = np.array([float(r["t4"] - r["t3"])
                                              for r in self.data[i_s:i_e]])

                    # Remove drift from observed timestamp differences
                    if (drift_comp):
                        t2_minus_t1_w = t2_minus_t1_w - drift_w
                        t4_minus_t3_w = t4_minus_t3_w + drift_w

                    if (strategy == 'median'):
                        x_est = self._sample_median(t2_minus_t1_w,
                                                    t4_minus_t3_w)

                    elif (strategy == 'min'):
                        x_est = self._eapf(t2_minus_t1_w,
                                           t4_minus_t3_w)

                    elif (strategy == 'max'):
                        x_est = self._sample_maximum(t2_minus_t1_w,
                                                     t4_minus_t3_w)

                    elif (strategy == 'mode'):
                        x_est = self._sample_mode(t2_minus_t1_w,
                                                  t4_minus_t3_w)
                    else:
                        raise ValueError("Strategy choice %s unknown" %(
                            strategy))

                # Re-add the total drift of this observation window to the
                # estimate that was produced by the selection operator
                if (drift_comp):
                    x_est = x_est + drift_w[-1]

                # Save on global data records
                self.data[i_e - 1]["x_pkts_{}".format(strategy)] = x_est

            if (strategy == 'mode' and
                (self._sample_mode_bin != SAMPLE_MODE_BIN_0)):
                logger.info("Sample-mode bin was increased up to %d" %(
                    self._sample_mode_bin))

