"""Analyse the estimators performance as a function of window length
"""
import argparse
import ptp.runner, ptp.reader, ptp.ls, ptp.pktselection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import re, json, time

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

class MaxTeVsWindow():
    def __init__(self, data, T_ns):
        """

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

    def _max_te_vs_window(self, data, estimator, plot=False):
        """Max|TE| vs window length

        Calculate the max|TE| for differents sizes of window length.

        Args:
            data      : Array of objects with simulation or testbed data
            estimator : Select the estimator
            plot      : Plot Max|TE| vs window

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
        last_print  = 0
        min_error   = np.inf
        i_iter      = 0
        count       = 0

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

            # The recursive moving average have transitories. Try to
            # skip it by throwing away an arbitrary number of initial values.
            self._sample_skip = 300 if (estimator == "sample-average") \
                                else self._sample_skip
            post_tran_data    = data[self._sample_skip:]

            # Get time offset estimation errors
            x_err = np.array([r[f"x_{est_key}"] - r["x"] for r in
                            post_tran_data if f"x_{est_key}" in r])
            # Erase results from runner data
            for r in data:
                r.pop(f"x_{est_key}", None)

            # Save max|TE|
            self.max_te[i] = np.amax(np.abs(x_err))

            # Break the code if the best window length is found, that is if the
            # window length that minimizes the Max|TE| persists for 80
            # consecutive windows.
            count += 1 if (min_error < self.max_te[i]) else count
            if (self.max_te[i] < min_error):
                min_error = self.max_te[i]
                count = 0
            if (count > 80):
                break

            # Save the index of the last iteration
            i_iter = i

        # Consider the max_te array only until the last iteration performed
        self.max_te = self.max_te[:i_iter]

        # Find the best window lenght and save the result
        i_best = np.argmin(self.max_te)
        N_best = window_len[i_best]
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

        print(f"Best evaluated window length for {est_name}: {N_best:d}")

    def _filename(self, file):
        """Format the filename correctly or create one if the file does not exist

        Args:
            file: Path of the JSON file

        Returns:
            The filename
        """
        path = "config/"

        if (file is None):
            filename = path + "runner-" + time.strftime("%Y%m%d-%H%M%S") + \
                       "-config" + ".json"
        else:
            filename = path + (re.search(r'([^//]*).json$', file).group(1)) + \
                       "-config" + ".json"

        return filename

    def _save(self, file):
        """Save est_op dictionary on JSON file

        Args:
            file : Path of the JSON file to save

        """
        filename = self._filename(file)
        with open(filename, 'w') as fd:
                json.dump(est_op, fd)

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
                sample_skip=0, window_skip=0, window_step=1):
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

        """
        self._sample_skip = sample_skip
        self._window_skip = window_skip
        self._window_step = window_step

        # Iterate over the estimators
        estimators = [k for k in est_op.keys()] if (estimator == 'all') \
                     else [estimator]

        for estimator in estimators:
            # For the sample filters estimators that require the drift
            # compensation provided by ls, first we need to find the best
            # window length for ls and then run it.
            if (re.search("-ls$", estimator)):
                if (est_op["ls"]["N_best"] is None):
                    self._max_te_vs_window(self.data, estimator="ls")

                ls = ptp.ls.Ls(est_op["ls"]["N_best"], self.data, self.T_ns)
                ls.process(impl="eff")

            # Run max_te_vs_window function
            self._max_te_vs_window(self.data, estimator=estimator, plot=plot)

        # Save results on JSON file
        if (save):
            self._save(file)