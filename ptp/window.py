"""Helper class used to optimize processing window lengths
"""
import logging, re, json, time
import ptp.ls, ptp.pktselection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


WINDOW_SEARCH_PATIENCE = 100 # used for early stopping

est_op = {"ls"            : {"name"   : "Least Squares",
                             "impl"   : "eff",
                             "est_key": "ls_eff",
                             "N_best" : None},
          "sample-average": {"name"   : "Sample Average",
                             "impl"   : "average",
                             "ls_impl": None,
                             "est_key": "pkts_average",
                             "N_best" : None},
          "sample-median" : {"name"   : "Sample Median",
                             "impl"   : "median",
                             "ls_impl": None,
                             "est_key": "pkts_median",
                             "N_best" : None},
          "sample-min"    : {"name"   : "Sample Minimum",
                             "impl"   : "min",
                             "ls_impl": None,
                             "est_key": "pkts_min",
                             "N_best" : None},
          "sample-min-ls" : {"name"   : "Sample Minimum with LS",
                             "impl"   : "min",
                             "ls_impl": "eff",
                             "est_key": "pkts_min_ls",
                             "N_best" : None},
          "sample-max"    : {"name"   : "Sample Maximum",
                             "impl"   : "max",
                             "ls_impl": None,
                             "est_key": "pkts_max",
                             "N_best" : None},
          "sample-mode"   : {"name"   : "Sample Mode",
                             "impl"   : "mode",
                             "ls_impl": None,
                             "est_key": "pkts_mode",
                             "N_best" : None},
          "sample-mode-ls": {"name"   : "Sample Mode with LS",
                             "impl"   : "mode",
                             "ls_impl": "eff",
                             "est_key": "pkts_mode_ls",
                             "N_best" : None}}


class Optimizer():
    def __init__(self, data, T_ns):
        """Optimizes processing window lengths

        Args:
            data: Array of objects with simulation or testbed data
            T_ns: Nominal message period in nanoseconds

        """
        self.data   = data
        self.T_ns   = T_ns

        # Max |TE| data
        self.max_te = None

        # Window configuration
        self._sample_skip = None
        self._window_step = None
        self._window_skip = None

    def _search_min_max_te(self, data, estimator, plot=False,
                           early_stopping=True):
        """Search the window length that minimizes Max|TE|

        Calculate the max|TE| for differents sizes of window length.

        Args:
            data           : Array of objects with simulation or testbed data
            estimator      : Select the estimator
            plot           : Plot Max|TE| vs window
            early_stopping : Whether to stop search when min{max|TE|} stalls

        """
        est_impl    = est_op[estimator]["impl"]
        est_key     = est_op[estimator]["est_key"]

        # Window lengths to evaluate
        window_len  = np.arange(2 + self._window_skip, int(len(data)/2), \
                                self._window_step)
        n_iter      = len(window_len)

        # Compute max|TE| for each window length
        self.max_te = np.zeros(window_len.shape)

        # Control variables
        last_print     = 0
        min_max_te     = np.inf
        i_iter         = 0
        patience_count = 0

        for i,N in enumerate(window_len):

            progress = (i/n_iter)
            if (progress - last_print > 0.1):
                print(f'{estimator} vs. window progress {progress*100:5.2f} %')
                last_print = progress

            if (estimator == "ls"):
                # Least-squares
                ls = ptp.ls.Ls(N, data, self.T_ns)
                ls.process(impl=est_impl)
            else:
                # Packet Selection
                ls_impl = est_op[estimator]["ls_impl"]
                pkts    = ptp.pktselection.PktSelection(N, data)
                pkts.process(strategy=est_impl, ls_impl=ls_impl)

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
            self.max_te[i] = np.amax(np.abs(x_err))

            # Keep track of minimum max|TE| with "early stopping"
            #
            # Stop search if the window length with minimum Max|TE| remains the
            # same for a number of consecutive windows and the difference
            # between the min{max|TE|} and the max|TE| of the current iteration
            # is higher than the min{|TE|}.
            #
            # NOTE: patience count tracks the number of iterations with no
            # reduction (or improvement) of max|TE|

            # Update min{max|TE|}
            if (self.max_te[i] < min_max_te):
                min_max_te = self.max_te[i] # min max|TE| so far
                N_best     = N              # best window length so far
                patience_count = 0
            else:
                patience_count += 1

            # Difference between the actual min{max|TE|} and the max|TE| of the
            # current iteration
            max_te_diff = abs(min_max_te - self.max_te[i])

            if (early_stopping and patience_count > WINDOW_SEARCH_PATIENCE \
               and max_te_diff > abs(min_max_te)):
                break

            # Save the index of the last iteration
            i_iter = i

        # Consider the max_te array only until the last evaluated iteration
        self.max_te = self.max_te[:i_iter]

        # Save the best window length
        est_op[estimator]["N_best"] = int(N_best)

        # Estimator name
        est_name = est_op[estimator]['name']

        if (plot):
            plt.figure()
            plt.scatter(window_len[self._window_skip:i_iter], \
                        self.max_te[self._window_skip:])
            plt.title(est_name)
            plt.xlabel("window length (samples)")
            plt.ylabel("max|TE| (ns)")
            plt.legend()
            plt.savefig(f"plots/{est_key}_max_te_vs_window")
            logging.info("Saved figure at %s" %(
                f"plots/{est_key}_max_te_vs_window"))

        print(f"Best evaluated window length for {est_name}: {N_best:d}")

    def _filename(self, file):
        """Create the filename, to save the est_op dictionary, based on the
        name of the file passed as argument or create one if no file was used.

        Args:
            file: Path of the file

        Returns:
            The filename
        """
        path = "config/"

        if (file is None):
            filename = path + "runner-" + time.strftime("%Y%m%d-%H%M%S") + \
                       "-config" + ".json"
        else:
            filename = path + (re.search(r'([^//]*).(json|npz)$', file).group(1)) \
                       + "-config" + ".json"

        return filename

    def _save(self, file):
        """Save est_op dictionary on JSON file

        Args:
            file : Path of the JSON file to save

        """
        filename = self._filename(file)
        with open(filename, 'w') as fd:
            json.dump(est_op, fd)

        logging.info("Saved window configurations on %s" %(filename))

    def load(self, file):
        """Load est_op from JSON file

        Args:
            file : Path of the JSON file to load

        """
        if (file):
            with open(file) as fd:
                est_op = json.load(fd)
        else:
            raise ValueError("Need to pass the filename to load the \
                             configuration data")

    def process(self, estimator, file=None, save=False, plot=False, \
                sample_skip=0, window_skip=0, window_step=1,
                early_stopping=True):
        """Process the observations

        Args:
            estimator       : Select the estimator
            file            : Path of the JSON file to save
            save            : Save the best window length in a json file
            plot            : Plot Max|TE| vs window
            sample_skip     : Number of initial samples to skip
            window_skip     : Number of initial windows to skip
            starting_window : Starting window size
            window_step     : Enlarge window by this step on every iteration
            early_stopping  : Whether to stop search when min{max|TE|} stalls

        """
        self._sample_skip = sample_skip
        self._window_skip = window_skip
        self._window_step = window_step

        # Iterate over the estimators
        estimators = [k for k in est_op.keys()] if (estimator == 'all') \
                     else [estimator]

        for estimator in estimators:
            # For the sample filters estimators that require the drift
            # compensation provided by LS, first we need to find the best window
            # length for LS and then run it.
            if (re.search("-ls$", estimator)):
                if (est_op["ls"]["N_best"] is None):
                    self._max_te_vs_window(self.data, estimator="ls")

                # Do we need to re-run?
                ls = ptp.ls.Ls(est_op["ls"]["N_best"], self.data, self.T_ns)
                ls.process()

            # Search the window length that minimizes the max|TE|
            self._search_min_max_te(self.data, estimator=estimator, plot=plot,
                                    early_stopping=early_stopping)

        # Save results on JSON file
        if (save):
            self._save(file)
