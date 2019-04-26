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

    def _sample_mean(self, x_obs):
        """Calculate the mean of a given time offset vector

        Args:
            x_obs   : Vector time offset

        Returns:
            The mean of the time offset vector
        """

        return np.mean(x_obs)

    def _sample_median(self, x_obs):
        """Calculate the median of a given time offset vector

        Args:
            x_obs   : Vector time offset

        Returns:
            The median of the time offset vector
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

    def process(self, strategy, ls_impl=None):
        """Process the observations

        Using the raw time offset measurements, estimate the time offset using
        sample-mean ("mean"), sample-median ("median") or sample-minimum ("min")
        over sliding windows of observations.

        Args:
            strategy  : Select the strategy of interest.
            ls_impl   : Apply packet selection on the time offset values fitted
            via LS using one of the three distinct implementations: "t2", "t1" and "eff".
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

        for i in range(0, (n_data - self.N) + 1):
            # Window start and end indexes
            i_s = i
            i_e = i + self.N

            # Observation window
            x_obs_w = x_obs[i_s:i_e]
            d_obs_w = d_obs[i_s:i_e]

            # Compute the time offset depending on the selected strategy
            if (strategy == 'mean'):
                x_est = self._sample_mean(x_obs_w)

            elif (strategy == 'median'):
                x_est = self._sample_median(x_obs_w)

            elif (strategy == 'min'):
                x_est = self._sample_minimum(x_obs_w, d_obs_w)

            else:
                raise ValueError("Strategy choice %s unknown" %(strategy))

            # Include Packet Selection estimations within the simulation data
            self.data[i_e - 1]["x_pkts_{}".format(strategy)] = x_est

