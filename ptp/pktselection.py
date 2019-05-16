import numpy as np

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

        Uses a ciruclar (ring) buffer with size 2*N, where the tail pointer is
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

    def _sample_median(self, x_obs):
        """Calculate the median of a given time offset vector

        Args:
            x_obs   : Vector time offset

        Returns:
            The moving average
        """

        return np.median(x_obs)

    def _sample_minimum(self, x_obs, d_obs):
        """Select the time offset corresponding to the minimum delay estimation

        Args:
            x_obs   : Vector time offset
            d_obs   : Vector delay estimation

        Returns:
           The time offset corresponding to the packet that has the smallest
           delay estimation.
        """
        i_d_est = d_obs.index(min(d_obs))
        return x_obs[i_d_est]

    def process(self, strategy, ls_impl=None, avg_impl="recursive"):
        """Process the observations

        Using the raw time offset measurements, estimate the time offset using
        sample-average ("average"), EWMA ("ewma"), sample-median ("median") or
        sample-minimum ("min") over sliding windows of observations.

        Args:
            strategy  : Select the strategy of interest.
            ls_impl   : Apply packet selection on the time offset values fitted
                        via LS using one of the three distinct implementations:
                        "t2", "t1" and "eff".
            avg_impl  : Sample-avg implementation ("recursive" or "normal").
                        The recursive implementation theoretically produces the
                        same result as the "normal" implementation, except
                        during initialization, due to the transitory of the
                        recursive implementation. However, the recursive one
                        is significantly more CPU efficient.

        """

        # Select vector of noisy time offset observations and delay estimation
        if (ls_impl):
            x_obs = [res["x_ls_{}".format(ls_impl)] for res in self.data if
                     "x_ls_{}".format(ls_impl) in res]
            d_obs = [res["d_est"] for res in self.data if
                     "x_ls_{}".format(ls_impl) in res]

            if (len(x_obs) <= 0 or len(d_obs) <= 0):
                raise ValueError("LS data not found")

        else:
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

                # Compute the time offset depending on the selected strategy
                if (strategy == 'average' and avg_impl == 'normal'):
                    x_est = self._sample_avg_normal(x_obs_w)
                elif (strategy == 'median'):
                    x_est = self._sample_median(x_obs_w)
                elif (strategy == 'min'):
                    x_est = self._sample_minimum(x_obs_w, d_obs_w)
                else:
                    raise ValueError("Strategy choice %s unknown" %(strategy))

                # Include Packet Selection estimations within the simulation data
                self.data[i_e - 1]["x_pkts_{}".format(strategy)] = x_est

