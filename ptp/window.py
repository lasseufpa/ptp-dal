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

    def _eval_max_te(self, window_vec, estimator, early_stopping=True,
                     patience=5):
        """Evaluate the max|TE| for a given estimator and given window lengths

        Args:
            window_vec     : Vector of window lengths to evaluate
            estimator      : Select the estimator
            early_stopping : Whether to stop search when min{max|TE|} stalls
            patience       : Number of consecutive iterations without
                             improvement to wait before signaling an early stop.

        Returns:
            N_best : Best evaluated window length
            max_te : vector with max|TE| computed for all given window lengths
            i_stop : Index where evaluation halted (if early stopping is active)

        """

        data      = self.data
        est_impl  = self.est_op[estimator]["impl"]
        est_key   = self.est_op[estimator]["est_key"]
        n_windows = len(window_vec)
        max_te    = np.zeros(n_windows)

        # Control variables
        last_print     = 0
        min_max_te     = np.inf
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

            # Compute max|TE| based on the entire "x_err" time series
            max_te[i] = np.amax(np.abs(x_err))

            # Keep track of minimum max|TE| with "early stopping"
            #
            # Stop search if the window length with minimum Max|TE| remains the
            # same for a number of consecutive windows.
            #
            # NOTE: patience count tracks the number of iterations with no
            # reduction (or improvement) of max|TE|

            # Update min{max|TE|}
            if (max_te[i] < min_max_te):
                min_max_te = max_te[i] # min max|TE| so far
                N_best     = N         # best window length so far
                patience_count = 0
            else:
                patience_count += 1

            if (early_stopping and patience_count > patience):
                break

            # Save the index of the last iteration
            i_iter = i

        return N_best, max_te, i_iter

    def _search_min_max_te(self, estimator, early_stopping=True, save=True,
                           plot=False, global_plot=False, plot_info=True,
                           fine_pass=False):
        """Search the window length that minimizes Max|TE|

        Calculate the max|TE| for differents sizes of window length. Runs two
        passes through the data. The first (coarse pass) evaluates power-of-2
        window lengths. The second (fine pass) evaluates intermediate values
        between the two best power-of-2 lengths.

        Args:
            estimator      : Select the estimator
            early_stopping : Whether to stop search when min{max|TE|} stalls
            save           : Save plot
            plot           : Plot Max|TE| vs window
            global_plot    : Plot global curve (not only the fine region)
            plot_info      : Add window information in the plot
            fine_pass      : Enable fine pass

        """

        est_key   = self.est_op[estimator]["est_key"]

        # Coarse pass
        #
        # Evaluate power-of-2 window lengths. If using early stopping, use the
        # default patience.
        log_max_window = np.floor(np.log2(len(self.data)/2))
        log_min_window = 1
        log_window_len = np.arange(log_min_window, log_max_window + 1, 1)
        window_len     = 2**log_window_len

        N_best, max_te, i_stop = self._eval_max_te(window_len, estimator,
                                                   early_stopping=early_stopping)

        # Truncate results by considering the early stopping index
        max_te     = max_te[:i_stop]
        i_max_te   = np.argsort(max_te[:i_stop])
        window_len = window_len[:i_stop]

        # Best and second best indexes
        i_best      = i_max_te[0]
        i_scnd_best = i_max_te[1]

        # Second best window length
        N_scnd_best = window_len[i_max_te[1]]

        # Before running the fine pass, prepare to concatenate previous max_te
        # values with the ones to be computed during the fine pass
        global_max_te  = max_te
        global_win_len = window_len

        if (fine_pass):
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

            N_best, max_te, i_stop = self._eval_max_te(window_len, estimator,
                                                       early_stopping=early_stopping,
                                                       patience=100)

            # Truncate results again by considering the early stopping index
            i_max_te   = np.argsort(max_te[:i_stop])
            max_te     = max_te[:i_stop]
            window_len = window_len[:i_stop]

            # Concatenate fine pass results within global vectors
            global_max_te  = np.concatenate((global_max_te, max_te))
            global_win_len = np.concatenate((global_win_len, window_len))

        # Save the best window length
        self.est_op[estimator]["N_best"] = int(N_best)

        # Estimator name
        est_name = self.est_op[estimator]['name']

        if (plot):
            plt.figure()
            if (global_plot):
                plt.scatter(global_win_len, global_max_te)
            else:
                plt.scatter(window_len[:i_stop], max_te[:i_stop])

            plt.title(est_name)
            plt.xlabel("window length (samples)")
            plt.ylabel("max|TE| (ns)")

            if (plot_info):
                plt.text(0.99, 0.98, f"Best window length: {N_best:d}",
                         transform=plt.gca().transAxes, va='top', ha='right')

            if (save):
                assert(self.plot_path is not None), "Plot path not defined"
                plt.savefig(os.path.join(self.plot_path,
                                         f"{est_key}_max_te_vs_window"))
            else:
                plt.show()

            logging.info("Saved figure at %s" %(
                f"plots/{est_key}_max_te_vs_window"))

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
            basename  = os.path.splitext(os.path.basename(file))[0]
            self.plot_path    = 'plots/' + basename + '/'
            self.cfg_filename = "config/" + basename + "-config" + ".json"
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

    def process(self, estimator, file=None, save=False, sample_skip=0,
                early_stopping=True, force=False, plot=False, save_plot=True,
                global_plot=False, plot_info=False, fine_pass=False):
        """Process the observations

        Args:
            estimator       : Select the estimator
            file            : Path of the JSON file to save
            save            : Save the best window length in a json file
            sample_skip     : Number of initial samples to skip
            starting_window : Starting window size
            early_stopping  : Whether to stop search when min{max|TE|} stalls
            force           : Force processing even if already done previously
            plot            : Plot Max|TE| vs window
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
            # Search the window length that minimizes the max|TE|
            if (not self.est_op[estimator]["N_best"]):
                self._search_min_max_te(estimator, plot=plot, save=save_plot,
                                        early_stopping=early_stopping,
                                        global_plot=global_plot,
                                        plot_info=plot_info,
                                        fine_pass=fine_pass)

            # Save results on JSON file
            if (save):
                self.save(file)
