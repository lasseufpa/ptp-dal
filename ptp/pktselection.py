import logging
import numpy as np
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

    def _sample_avg_normal(self, x_obs):
        """Calculate the average of a given time offset vector

        Args:
            x_obs   : Vector of time offset observations

        Returns:
            The average of the time offset vector
        """

        return np.mean(x_obs)

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
        """Calculate the median of a given time offset vector

        Args:
            t2_minus_t1_w : Vector of (t2 - t1) differences
            t4_minus_t3_w : Vector of (t4 - t3) differences

        Returns:
            The moving average
        """

        t2_minus_t1 = np.median(t2_minus_t1_w)
        t4_minus_t3 = np.median(t4_minus_t3_w)
        x_est       = (t2_minus_t1 - t4_minus_t3)/2
        return x_est

    def _eapf(self, t2_minus_t1_w, t4_minus_t3_w, drift=None):
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

        If vector of drifts is defined, use it to compensate the differences of
        timestamps along the observation window. Note that:

            x[n] = t2[n] - (t1[n] + dms[n])
        or
            x[n] = (t3[n] + dsm[n]) - t4[n]

        so that

            t2[n] - t1[n] = +x[n] + dms[n],
            t4[n] - t3[n] = -x[n] + dsm[n]

        However, x[n] can be modeled as:

            x[n] = x_0 + drift[n]

        If the drift is compensated, we have:

            t2[n] - t1[n] - drift[n] = +x_0 + dms[n],
            t4[n] - t3[n] + drift[n] = -x_0 + dsm[n]

        In the end, the conventional sample-minumum results in:

            x_est -> x_0

        Thus, a final dirft compensation must be applied:

            x_est += drift[N-1]

        Args:
            t2_minus_t1_w : Vector of (t2 - t1) differences
            t4_minus_t3_w : Vector of (t4 - t3) differences
            drift         : Vector of time drifts accumulated along the window

        Returns:
           The time offset estimate

        """
        # Subtract the drift from slave timestamps
        if (drift is not None):
            t2_minus_t1_w = t2_minus_t1_w - drift
            t4_minus_t3_w = t4_minus_t3_w + drift
            offset        = drift[-1]
        else:
            offset = 0

        t2_minus_t1 = np.amin(t2_minus_t1_w)
        t4_minus_t3 = np.amin(t4_minus_t3_w)

        # Final estimate (with drift compensation offset)
        x_est       = offset + (t2_minus_t1 - t4_minus_t3)/2

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

    def _sample_mode(self, t2_minus_t1_w, t4_minus_t3_w, drift=None,
                     cnt_threshold=3, stall_patience=10):
        """Compute the time offset based on sample-mode

        For the drift compensation, c.f. the explanation of EAPF.

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

        # Subtract the drift from slave timestamps
        if (drift is not None):
            t2_minus_t1_w = t2_minus_t1_w - drift
            t4_minus_t3_w = t4_minus_t3_w + drift
            offset        = drift[-1]
        else:
            offset = 0

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

        # Final estimate (with drift compensation offset)
        x_est            = offset + (t2_minus_t1 - t4_minus_t3)/2
        return x_est

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

    def process(self, strategy, ls_impl=None, avg_impl="recursive"):
        """Process the observations

        Using the raw time offset measurements, estimate the time offset using
        sample-average ("average"), EWMA ("ewma"), sample-median ("median") or
        sample-minimum ("min") over sliding windows of observations.

        Args:
            strategy  : Select the strategy of interest.
            ls_impl   : Combine packet selection to information/estimations
                        obtained via LS using one of the three distinct
                        implementations: "t2", "t1" and "eff". The specific
                        combination is defined per selection method.
            avg_impl  : Sample-avg implementation ("recursive" or "normal").
                        The recursive implementation theoretically produces the
                        same result as the "normal" implementation, except
                        during initialization, due to the transitory of the
                        recursive implementation. However, the recursive one
                        is significantly more CPU efficient.

        """

        logger.info("Processing sample-%s with N=%d" %(strategy, self.N))

        # Select vector of noisy time offset observations and delay estimation
        x_obs = [res["x_est"] for res in self.data]
        d_obs = [res["d_est"] for res in self.data]

        n_data = len(x_obs)

        if (strategy == 'average' and avg_impl == 'recursive'):
            # Sample-by-sample processing
            for i in range(0, n_data):
                x_est = self._sample_avg_recursive(x_obs[i])
                self.data[i]["x_pkts_average"] = x_est
        elif (strategy == 'ewma'):
            # Sample-by-sample processing
            for i in range(0, n_data):
                x_est = self._sample_avg_ewma(x_obs[i])
                self.data[i]["x_pkts_ewma"] = x_est
        else:
            # Window-based processing
            for i in range(0, (n_data - self.N) + 1):
                # Window start and end indexes
                i_s = i
                i_e = i + self.N

                # Observation window
                x_obs_w = x_obs[i_s:i_e]
                d_obs_w = d_obs[i_s:i_e]

                # Apply frequency compensation using the frequency offset
                # estimation obtained via LS
                if (ls_impl is not None and
                    "y_ls_{}".format(ls_impl) in self.data[i_e - 1]):

                    # Compute the drift within the observation window
                    t_w   = np.array([float(r["t1"] - self.data[i_s]["t1"]) for
                                      r in self.data[i_s:i_e]])
                    y     = self.data[i_e - 1]["y_ls_{}".format(ls_impl)]
                    drift = y * t_w
                else:
                    drift = None

                # Selection operator that processes observations directly
                if (strategy == 'average' and avg_impl == 'normal'):
                    x_est = self._sample_avg_normal(x_obs_w)

                # Selection operator that processes timestamp differences
                else:
                    t2_minus_t1_w = np.array([float(r["t2"] - r["t1"])
                                              for r in self.data[i_s:i_e]])
                    t4_minus_t3_w = np.array([float(r["t4"] - r["t3"])
                                              for r in self.data[i_s:i_e]])

                    if (strategy == 'median'):
                        x_est = self._sample_median(t2_minus_t1_w,
                                                    t4_minus_t3_w)

                    elif (strategy == 'min'):
                        x_est = self._eapf(t2_minus_t1_w,
                                           t4_minus_t3_w,
                                           drift)

                    elif (strategy == 'max'):
                        x_est = self._sample_maximum(t2_minus_t1_w,
                                                     t4_minus_t3_w)

                    elif (strategy == 'mode'):
                        x_est = self._sample_mode(t2_minus_t1_w,
                                                  t4_minus_t3_w,
                                                  drift)
                    else:
                        raise ValueError("Strategy choice %s unknown" %(
                            strategy))

                # Include Packet Selection estimations within the simulation data
                if (drift is not None):
                    self.data[i_e - 1]["x_pkts_{}_ls".format(strategy)] = x_est
                else:
                    self.data[i_e - 1]["x_pkts_{}".format(strategy)] = x_est

            if (strategy == 'mode' and
                (self._sample_mode_bin != SAMPLE_MODE_BIN_0)):
                logger.info("Sample-mode bin was increased up to %d" %(
                    self._sample_mode_bin))

