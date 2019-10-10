"""Helper class used to optimize processing window lengths
"""
import logging, re, json, time, os
import ptp.ls, ptp.pktselection
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
                           "impl"   : "average",
                           "est_key": "pkts_average",
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

    def __init__(self, data, T_ns):
        """Optimizes processing window lengths

        Args:
            data: Array of objects with simulation or testbed data
            T_ns: Nominal message period in nanoseconds

        """
        self.data   = data
        self.T_ns   = T_ns

        # Window configuration
        self._sample_skip = None

        self.cfg_filename = None
        self.plot_path    = None

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
            error  : vector with the error computed for all given window lengths
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
                ls.process(impl=est_impl)
            else:
                pkts    = ptp.pktselection.PktSelection(N, data)
                pkts.process(strategy=est_impl)

            # The recursive moving average methods have transitories. Try to
            # skip them by throwing away an arbitrary number of initial values.
            self._sample_skip = 300 if (estimator == "sample-average") \
                                else self._sample_skip
            post_tran_data    = data[self._sample_skip:]

            # Get time offset estimation errors
            x_err = np.array([r[f"x_{est_key}"] - r["x"] for r in
                              post_tran_data if f"x_{est_key}" in r])

            # Erase results from runner data
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
                            save=True, plot=False, global_plot=False,
                            plot_info=True, fine_pass=False, eval_all=False):
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
            save           : Save plot
            plot           : Plot error vs window
            global_plot    : Plot global curve (not only the fine region)
            plot_info      : Add window information in the plot
            fine_pass      : Enable fine pass
            eval_all       : Disable coarse/fine pass and instead evaluate all
                             window length values of a linear range

        """
        est_key   = self.est_op[estimator]["est_key"]

        if (eval_all):
            max_window = 2**np.minimum(np.floor(np.log2(len(self.data)/2)), 13)
            window_len = np.arange(2, max_window)
            N_best, error , i_stop = self._eval_error(window_len, estimator,
                                                      error_metric=error_metric,
                                                      early_stopping=False)
        else:
            # Coarse pass
            #
            # Evaluate power-of-2 window lengths. If using early stopping, use
            # the default patience.
            log_max_window = np.minimum(np.floor(np.log2(len(self.data)/2)), 13)
            # limit to 8192
            if (est_key == "pkts_mode"):
                log_min_window = 2 # sample-mode needs window length > 2
            else:
                log_min_window = 1
            log_window_len = np.arange(log_min_window, log_max_window + 1, 1)
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
            # Evaluate window lengths between the two best power-of-2 window lengths
            # of the coarse pass. If using early stopping, use a relative high
            # patience for the fine pass, as this region of the curve can be noisy.

            # Sanity check
            if (np.abs(i_scnd_best - i_best) != 1):
                logging.warning("Best (%d) and second-best (%d) windows are not \
                consecutive" %(N_best, N_scnd_best))

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

        # Save the best window length
        self.est_op[estimator]["N_best"] = int(N_best)

        # Estimator name
        est_name = self.est_op[estimator]['name']

        if (plot):
            plt.figure()
            if (global_plot):
                plt.scatter(global_win_len, global_error)
            else:
                plt.scatter(window_len[:i_stop], error[:i_stop])

            plt.title(est_name)
            plt.xlabel("window length (samples)")
            plt.ylabel(f"{error_metric} (ns)")

            if (plot_info):
                plt.text(0.99, 0.98, f"Best window length: {N_best:d}",
                         transform=plt.gca().transAxes, va='top', ha='right')

            if (save):
                assert(self.plot_path is not None), "Plot path not defined"
                plt.savefig(os.path.join(self.plot_path,
                                         f"win_opt_{est_key}_{error_metric}_error_vs_window"),
                            dpi=300)
                logging.info("Saved figure at %s" %(
                    f"{self.plot_path}/{est_key}_{error_metric}_vs_window"))
            else:
                plt.show()


        logger.info(f"Best evaluated window length for {est_name}: {N_best:d}")

    def _set_paths(self, file):
        """Define paths to save plots and configuration file

        Create a folder with the name of the file used to generate the metrics
        and save all the plots inside it, or if no file is used just save
        the metrics within the folder 'plots/'.

        Set the config filename based on the name of the file passed as argument
        or create a new name if no file was used.

        Args:
            file : Path of the file

        """
        if (file):
            # Aim at saving plots at "rootdir/plots/" and window configurations
            # at "rootdir/config". The current module lies within
            # "rootdir/ptp/", so we use its path in order to find rootdir.
            basename  = os.path.splitext(os.path.basename(file))[0]
            this_file = os.path.realpath(__file__)
            rootdir   = os.path.dirname(os.path.dirname(this_file))
            self.plot_path    = os.path.join(rootdir, 'plots', basename)
            self.cfg_filename = os.path.join(rootdir, 'config',
                                             basename + "-config" + ".json")
        else:
            self.plot_path    = 'plots/'
            self.cfg_filename = "config/runner-" + \
                                time.strftime("%Y%m%d-%H%M%S") + "-config" + ".json"

        # Create the folder if it doesn't exist
        if not os.path.isdir(self.plot_path):
            os.makedirs(self.plot_path)

    def save(self):
        """Save est_op dictionary on JSON file

        """
        assert(self.cfg_filename is not None), \
            "Configuration filename not defined"

        with open(self.cfg_filename, 'w') as fd:
            json.dump(self.est_op, fd)

        logging.info("Saved window configurations on %s" %(self.cfg_filename))

    def load(self, file):
        """Load est_op from JSON file

        Args:
            file : Path of the JSON file to load

        """
        if (file):
            with open(file) as fd:
                self.est_op = json.load(fd)
        else:
            raise ValueError("Need to pass the filename to load the \
                             configuration data")

    def _is_cfg_file_complete(self):
        """Check if configuration file is complete, i.e. if it contains the
        best window for all possible estimators.

        Return:
            bool  : Returns 'True' if file is complete.

        """
        assert(self.cfg_filename is not None)
        cfg_file = self.cfg_filename
        logger.info("Loading configurations from %s." %(cfg_file))
        self.load(cfg_file)

        # Check if the file is complete, i.e. have window configuration
        # of all estimators
        for k, v in self.est_op.items():
            if (not v["N_best"]):
                print("Window configuration file is incomplete.")
                raw_resp = input("Find missing windows? [Y/n] ") or "Y"
                response = raw_resp.lower()
                if (response == 'y'):
                    return True
                else:
                    break
        return False

    def process(self, estimator, error_metric="max-te", file=None, save=False,
                sample_skip=0, early_stopping=True, force=False, plot=False,
                save_plot=True, global_plot=False, plot_info=False,
                fine_pass=False):
        """Process the observations

        Args:
            estimator       : Select the estimator
            error_metric    : Estimation errror metric (Max|TE| or MSE)
            file            : Path of the JSON file to save
            save            : Save the best window length in a json file
            sample_skip     : Number of initial samples to skip
            starting_window : Starting window size
            early_stopping  : Whether to stop search when min{error} stalls
            force           : Force processing even if already done previously
            plot            : Plot error vs window
            save_plot       : Save plot if plotting
            global_plot     : Plot global curve (not only the fine region)
            plot_info       : Add window information in the plot
            fine_pass       : Enable fine pass

        """
        self._sample_skip = sample_skip

        # Define paths used for saving plots and config file
        self._set_paths(file)

        # Is there a configuration file already? Is it complete (with all
        # information)?
        if (os.path.isfile(self.cfg_filename)):
            logger.info("Window tuning file %s exists." %(self.cfg_filename))
            if (force):
                logger.info("Cleaning configurations from %s." %(
                    self.cfg_filename))
                self.save()
            else:
                if (self._is_cfg_file_complete()):
                    return

        # Iterate over the estimators
        estimators = [k for k in self.est_op.keys()] if (estimator == 'all') \
                     else [estimator]

        for estimator in estimators:
            # Search the window length that minimizes the calculated error
            if (not self.est_op[estimator]["N_best"] or force):
                self._search_best_window(estimator, error_metric=error_metric,
                                         plot=plot, save=save_plot,
                                         early_stopping=early_stopping,
                                         global_plot=global_plot,
                                         plot_info=plot_info,
                                         fine_pass=fine_pass)

            # Save results on JSON file
            if (save):
                self.save()
