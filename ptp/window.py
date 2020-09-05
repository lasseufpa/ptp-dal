"""Helper class used to optimize processing window lengths
"""
import logging, re, json, time, os
import ptp.ls, ptp.pktselection, ptp.cache, ptp.bias
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)


class Optimizer():
    est_op = {
        "ls"            : {"name"   : "Least Squares",
                           "impl"   : "eff",
                           "est_key": "ls_eff",
                           "N_best" : None},
        "sample-average": {"name"   : "Sample Average",
                           "impl"   : "avg-recursive",
                           "est_key": "pkts_avg_recursive",
                           "N_best" : None},
        "sample-ewma"   : {"name"   : "EWMA",
                           "impl"   : "ewma",
                           "est_key": "pkts_ewma",
                           "N_best" : None},
        "sample-median" : {"name"   : "Sample Median",
                           "impl"   : "median",
                           "est_key": "pkts_median",
                           "N_best" : None},
        "sample-min"    : {"name"   : "Sample Minimum",
                           "impl"   : "min",
                           "est_key": "pkts_min",
                           "N_best" : None},
        "sample-max"    : {"name"   : "Sample Maximum",
                           "impl"   : "max",
                           "est_key": "pkts_max",
                           "N_best" : None},
        "sample-mode"   : {"name"   : "Sample Mode",
                           "impl"   : "mode",
                           "est_key": "pkts_mode",
                           "N_best" : None}
    }

    def __init__(self, data, T_ns, opts):
        """Optimizes processing window lengths

        Args:
            data : Array of objects with simulation or testbed data
            T_ns : Nominal message period in nanoseconds
            opts : Packet selection or LS algorithm options


        """
        self.data = data
        self.T_ns = T_ns
        self.opts = opts

        # Number of samples
        self.n_data   = len(data)

        # Window configuration
        self._sample_skip = None

    def _correct_pkts_bias(self, key):
        """Correct the bias of packet selection algorithms"""
        bias_corr_mode = self.opts['bias_corr_mode']
        bias_est       = self.opts['bias_est']

        if (not (bias_corr_mode == 'post' or bias_corr_mode == 'both')):
            return

        bias = ptp.bias.Bias(self.data)
        for metric in ['median', 'min', 'max', 'mode']:
            if (metric in bias_est and bias_est[metric]):
                bias.compensate(corr=bias_est[metric],
                                toffset_key=f"x_pkts_{metric}")
            else:
                logging.warning(f"Can't compensate asymmetry of {metric}")

    def _eval_error(self, window_vec, estimator, error_metric,
                    early_stopping=True, patience=5):
        """Evaluate the error for a given estimator and given window lengths

        Args:
            window_vec     : Vector of window lengths to evaluate
            estimator      : Select the estimator
            error_metric   : Chosen error metric (Max|TE| or MSE)
            early_stopping : Whether to stop search when min{error} stalls
            patience       : Number of consecutive iterations without
                             improvement to wait before signaling an early stop.

        Returns:
            N_best : Best evaluated window length
            error  : Vector with the error computed for all given window lengths
            i_stop : Index where evaluation halted (if early stopping is active)

        """
        data      = self.data
        est_impl  = self.est_op[estimator]["impl"]
        est_key   = self.est_op[estimator]["est_key"]
        n_windows = len(window_vec)
        error     = np.zeros(n_windows)

        # Control variables
        last_print     = 0
        min_err        = np.inf
        i_iter         = 0
        patience_count = 0

        for i,N in enumerate(window_vec):
            N = int(N)

            # Track progress
            progress = (i/n_windows)
            if (progress - last_print > 0.1):
                logger.info(f'{estimator} vs. window progress {progress*100:5.2f} %')
                last_print = progress

            # Run estimator
            if (estimator == "ls"):
                ls = ptp.ls.Ls(N, data, self.T_ns)
                ls.process(impl=est_impl,
                           batch_size=self.opts['batch_size'])
            else:
                pkts = ptp.pktselection.PktSelection(N, data)
                pkts.process(strategy=est_impl,
                             drift_comp=self.opts['drift_comp'],
                             batch_size=self.opts['batch_size'])

            # Correct bias if bias correction is enabled for packet selection
            # algorithms
            #
            # NOTE: while packet selection algorithms use the bias
            # post-compensation strategy (bias corrected on algorithm's output),
            # LS can rely on the pre-compensation strategy (bias corrected on
            # algorithms' input). Hence, if the goal is to apply bias correction
            # on LS, it should be applied before calling the window
            # optimizer. Refer to the implementation on analyze.py.
            #
            # Furthermore, note that bias compensation is important because the
            # window optimizer must observe the same performance that will be
            # obtained with the final processing of the algorithm (processed
            # with the optimal window size). If bias correction is enabled for
            # the final processing, then it must be considered here too.
            #
            # If the bias correction stage was ignored here, and if the bias is
            # more significant than the variance, the optimizer would tend to
            # converge to window sizes that coincidentally lead to the lowest
            # bias. For example, if drift correction is not enabled, a specific
            # window size may tend to accumulate enough time offset drift to
            # counteract the algorithm's bias.
            if (estimator != "ls"):
                self._correct_pkts_bias(est_key)

            # The recursive moving average methods have transitories. Try to
            # skip them by throwing away an arbitrary number of initial values.
            self._sample_skip = 300 if (estimator == "sample-average") \
                                else self._sample_skip
            post_tran_data    = data[self._sample_skip:]

            # Get time offset estimation errors
            x_err = np.array([r[f"x_{est_key}"] - r["x"] for r in
                              post_tran_data if f"x_{est_key}" in r])

            # Erase results from dataset
            for r in data:
                r.pop(f"x_{est_key}", None)

            # Compute error based on different metrics. Note that the entire
            # "x_err" time series are used to compute the final error.
            if (error_metric == "max-te"):
                error[i] = np.amax(np.abs(x_err))
            elif (error_metric == "mse"):
                error[i] = np.square(x_err).mean()
            else:
                raise ValueError("Error metric %s is invalid" %(error_metric))

            # Keep track of minimum error with "early stopping"
            #
            # Stop search if the window length with minimum error remains the
            # same for a number of consecutive windows.
            #
            # NOTE: patience count tracks the number of iterations with no
            # error reduction (or improvement).

            # Update min{error}
            if (error[i] < min_err):
                min_err = error[i] # min error so far
                N_best  = N        # best window length so far
                patience_count = 0
            else:
                patience_count += 1

            if (early_stopping and patience_count > patience):
                break

            # Save the index of the last iteration
            i_iter = i

        return N_best, error, i_iter

    def _search_best_window(self, estimator, error_metric, early_stopping=True,
                            fine_pass=False, eval_all=False, log_max_window=13,
                            save_global=False):
        """Search the best window length that minimizes error

        Calculate the error for differents sizes of window length. Runs two
        passes through the data. The first (coarse pass) evaluates power-of-2
        window lengths. The second (fine pass) evaluates intermediate values
        between the two best power-of-2 lengths.

        If option "eval_all" is defined, disable coarse-fine passes and instead
        run a normal pass over a range with unitary increments. Say this range
        will be 2:2048. In this case, evaluate all values in this range.


        Args:
            estimator      : Select the estimator
            error_metric   : The error metric (max-te or mse)
            early_stopping : Whether to stop search when min{error} stalls
            fine_pass      : Enable fine pass
            eval_all       : Disable coarse/fine pass and instead evaluate all
                             window length values of a linear range
            log_max_window : Log2 of the upper limit set for window
                             lengths.
            save_global    : Save global curve (not only the fine region) on
                             cache file

        """
        est_key  = self.est_op[estimator]['est_key']
        est_name = self.est_op[estimator]['name']

        if (eval_all):
            end_window = 2**np.minimum(np.floor(np.log2(len(self.data)/2)),
                                       log_max_window)
            window_len = np.arange(2, end_window)
            N_best, error , i_stop = self._eval_error(window_len, estimator,
                                                      error_metric=error_metric,
                                                      early_stopping=False)
        else:
            # Coarse pass
            #
            # Evaluate power-of-2 window lengths. If using early stopping, use
            # the default patience.
            log_len_e = np.minimum(np.floor(np.log2(len(self.data)/2)),
                                   log_max_window)
            if (est_key == "pkts_mode"):
                log_len_s = 2 # sample-mode needs window length > 2
            else:
                log_len_s = 1
            log_window_len = np.arange(log_len_s, log_len_e + 1, 1)
            window_len     = 2**log_window_len

            N_best, error, i_stop = self._eval_error(window_len, estimator,
                                                     error_metric=error_metric,
                                                     early_stopping=early_stopping)

        # Truncate results by considering the early stopping index
        error      = error[:i_stop]
        i_error    = np.argsort(error[:i_stop])
        window_len = window_len[:i_stop]

        # Best and second best indexes
        i_best      = i_error[0]
        i_scnd_best = i_error[1]

        # Second best window length
        N_scnd_best = window_len[i_error[1]]

        # Before running the fine pass, prepare to concatenate previous error
        # values with the ones to be computed during the fine pass
        global_error   = error
        global_win_len = window_len

        if (fine_pass and (not eval_all)):
            # Fine pass
            #
            # Evaluate window lengths between the two best power-of-2 window
            # lengths of the coarse pass. If using early stopping, use a
            # relative high patience for the fine pass, as this region of the
            # curve can be noisy.

            # Sanity check
            if (np.abs(i_scnd_best - i_best) != 1):
                logger.warning("Best (%d) and second-best (%d) windows are not "
                               "consecutive" %(N_best, N_scnd_best))

            # Define the fine range of window lengths and run
            if (N_best > N_scnd_best):
                window_len = np.arange(N_scnd_best, N_best, 1)
            else:
                window_len = np.arange(N_best, N_scnd_best, 1)

            N_best, error, i_stop = self._eval_error(window_len, estimator,
                                                     error_metric=error_metric,
                                                     early_stopping=early_stopping,
                                                     patience=100)

            # Truncate results again by considering the early stopping index
            i_error    = np.argsort(error[:i_stop])
            error      = error[:i_stop]
            window_len = window_len[:i_stop]

            # Concatenate fine pass results within global vectors
            global_error   = np.concatenate((global_error, error))
            global_win_len = np.concatenate((global_win_len, window_len))

        if (save_global):
            plot_window_len = global_win_len
            plot_error      = global_error
        else:
            plot_window_len = window_len[:i_stop]
            plot_error      = error[:i_stop]

        # Save tunning information
        self.est_op[estimator]["N_best"]       = int(N_best)
        self.est_op[estimator]["window_len"]   = plot_window_len.tolist()
        self.est_op[estimator]["window_error"] = plot_error.tolist()
        self.est_op[estimator]["error_metric"] = error_metric
        self.est_op[estimator]["n_samples"]    = self.n_data

        logger.info(f"Best evaluated window length for {est_name}: {N_best:d}")

    def _is_cache_complete(self, data, metric):
        """Check if the cached configuration is complete

        The cache file is considered complete if it contains the optimal window
        for all estimators computed based on the chosen error metric.

        Args:
            data   : Cached optimal windows configuration
            metric : Error metric (Max|TE| or MSE)

        Return:
            (bool) Whether the cached data is complete.

        """
        for v in data.values():
            if (v["N_best"] is None or \
                v["error_metric"] != metric or \
                v["n_samples"] != self.n_data):

                logger.warning("Window configuration cache file is incomplete.")
                return False

        return True

    def process(self, estimator, error_metric="max-te", cache=None,
                sample_skip=0, early_stopping=True, force=False,
                fine_pass=False, max_window=8192, save_global=False):
        """Process the observations

        Args:
            estimator       : Select the estimator
            error_metric    : Estimation error metric (Max|TE| or MSE)
            cache           : Cache handler used to save the optimized
                              configuration in a json file
            sample_skip     : Number of initial samples to skip
            early_stopping  : Whether to stop search when min{error} stalls
            force           : Force processing even if already done previously
            fine_pass       : Enable fine pass
            max_window      : Upper limit set for window length. This is mostly
                              to prevent excessive memory usage and slow
                              processing that occurs with very long windows.
            save_global     : Save global curve (not only the fine region) on
                              the cache file

        """
        self._sample_skip = sample_skip

        # Power-of-2 maximum window length
        log_max_window = np.floor(np.log2(max_window))
        if ((2**log_max_window) != max_window):
            logger.warning("Max window length set to {} instead of {}".format(
                (2**log_max_window), max_window))

        # Is there a configuration file already? Is it complete (with all
        # information)?
        if (cache is not None):
            assert(isinstance(cache, ptp.cache.Cache)), "Invalid cache object"
            cached_cfg = cache.load('window')

            if (cached_cfg and not force):
                self.est_op = cached_cfg
                logger.info("Found cached optimal window configurations")
                if (self._is_cache_complete(cached_cfg, error_metric)):
                    return
        else:
            logger.info("Unable to find existed configuration file")

        # Estimators to optimize
        if (estimator == 'all'):
            # All estimators, except the ones that are already optimized (within
            # the results that were loaded from cache)
            estimators = [k for k in self.est_op.keys() if
                          (self.est_op[k]["N_best"] is None or
                           self.est_op[k]["error_metric"] != error_metric or
                           self.est_op[k]["n_samples"] != self.n_data)]
        else:
            estimators = [estimator]

        for estimator in estimators:
            # Search the window length that minimizes the calculated error
            self._search_best_window(estimator, error_metric=error_metric,
                                     early_stopping=early_stopping,
                                     fine_pass=fine_pass,
                                     log_max_window=log_max_window,
                                     save_global=save_global)

            # Save results on JSON file
            if (cache is not None):
                cache.save(self.est_op, identifier='window')

    def get_results(self):
        """Get the best window for each post-processing method

        Returns:
            Dictionary with best window lengths for each method

        """
        return {
            'ls'      : self.est_op["ls"]["N_best"],
            'movavg'  : self.est_op["sample-average"]["N_best"],
            'median'  : self.est_op["sample-median"]["N_best"],
            'min'     : self.est_op["sample-min"]["N_best"],
            'max'     : self.est_op["sample-max"]["N_best"],
            'mode'    : self.est_op["sample-mode"]["N_best"],
            'ewma'    : self.est_op["sample-ewma"]["N_best"]
        }

    def print_results(self):
        """Print window length results"""

        print("Tuned window lengths:")
        for i in self.est_op:
            print("%20s: %d" %(i, self.est_op[i]["N_best"]))

