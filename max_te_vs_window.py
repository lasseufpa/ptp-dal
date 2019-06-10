"""Analyse the estimators performance as a function of window length
"""
import argparse
import ptp.runner, ptp.reader, ptp.ls, ptp.pktselection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import re

est_op = {"ls"            : {"title"  : "Least Squares",
                             "impl"   : "eff",
                             "est_key": "ls_eff",
                             "N_best" : None},
          "sample-average": {"title"  : "Sample Average",
                             "impl"   : "average",
                             "ls_impl": None,
                             "est_key": "pkts_average",
                             "N_best" : None},
          "sample-median" : {"title"  : "Sample Median",
                             "impl"   : "median",
                             "ls_impl": None,
                             "est_key": "pkts_median",
                             "N_best" : None},
          "sample-min"    : {"title"  : "Sample Minimum",
                             "impl"   : "min",
                             "ls_impl": None,
                             "est_key": "pkts_min",
                             "N_best" : None},
          "sample-min-ls" : {"title"  : "Sample Minimum with LS",
                             "impl"   : "min",
                             "ls_impl": "eff",
                             "est_key": "pkts_min_ls",
                             "N_best" : None},
          "sample-max"    : {"title"  : "Sample Maximum",
                             "impl"   : "max",
                             "ls_impl": None,
                             "est_key": "pkts_max",
                             "N_best" : None},
          "sample-mode"   : {"title"  : "Sample Mode",
                             "impl"   : "mode",
                             "ls_impl": None,
                             "est_key": "pkts_mode",
                             "N_best" : None},
          "sample-mode-ls": {"title"  : "Sample Mode with LS",
                             "impl"   : "mode",
                             "ls_impl": "eff",
                             "est_key": "pkts_mode_ls",
                             "N_best" : None}}

def max_te_vs_window(data, estimator, n_skip=0):
    """Max|TE| vs window length

    Calculate the max|TE| for differents sizes of window length.

    Args:
        data      : Array of objects with simulation or testbed data
        estimator : Select the estimator
        n_skip    : Number of initial samples to skip

    """
    est_impl = est_op[estimator]["impl"]
    est_key  = est_op[estimator]["est_key"]

    # Window lengths to evaluate:
    window_len = np.arange(2, int(len(data)/2))
    n_iter     = len(window_len)

    # Compute max|TE| for each window length
    max_te     = np.zeros(window_len.shape)
    last_print = 0

    # Control variables
    min_error  = np.inf
    i_iter     = 0
    count      = 0

    for i,N in enumerate(window_len):

        progress = (i/n_iter)
        if (progress - last_print > 0.01):
            print(f'{estimator} vs. window progress {progress*100:5.2f} %')
            last_print = progress

        if (estimator == "ls"):
            ls   = ptp.ls.Ls(N, data)
            ls.process(impl=est_impl)
        else:
            ls_impl = est_op[estimator]["ls_impl"]
            pkts    = ptp.pktselection.PktSelection(N, data)
            pkts.process(strategy=est_impl, ls_impl=ls_impl)

        # The recursive moving average have transitories. Try to
        # skip it by throwing away an arbitrary number of initial values.
        n_skip = 300 if (estimator == "sample-average") else n_skip
        post_tran_data = data[n_skip:]

        # Get time offset estimation errors
        x_err = np.array([r[f"x_{est_key}"] - r["x"] for r in
                          post_tran_data if f"x_{est_key}" in r])
        # Erase results from runner data
        for r in data:
            r.pop(f"x_{est_key}", None)

        # Save max|TE|
        max_te[i] = np.amax(np.abs(x_err))

        # Break the code if the best window length is found, that is if the
        # window length that minimizes the Max|TE| persists for 80 consecutive
        # windows.
        count += 1 if (min_error < max_te[i]) else count
        if (max_te[i] < min_error):
            min_error = max_te[i]
            count = 0
        if (count > 80):
            i_iter = i      # Save the index of the last iteration so that the
            break           # calculation of the error is considered until here.

    # Find the best window lenght and save the result
    i_best = np.argmin(max_te[:i_iter])
    N_best = window_len[i_best]
    est_op[estimator]["N_best"] = N_best

    plt.figure()
    plt.scatter(window_len[:i_iter], max_te[:i_iter])
    plt.title(f"{est_op[estimator]['title']}")
    plt.xlabel("window length (samples)")
    plt.ylabel("max|TE| (ns)")
    plt.legend()
    plt.savefig(f"plots/{est_key}_max_te_vs_window")

    print(f"Best evaluated window length: {N_best:d}")

def main():
    # Available estimators
    est_choices = [k for k in est_op] + ['all']

    parser = argparse.ArgumentParser(description="Max|TE| vs window")
    parser.add_argument('-e', '--estimator',
                        default='all',
                        help='Window-based estimator',
                        choices=est_choices)
    parser.add_argument('-f', '--file',
                        default=None,
                        help='Serial capture file.')
    args = parser.parse_args()

    if (args.file is None):
        # Run PTP simulation
        n_iter  = 20000
        ptp_src = ptp.runner.Runner(n_iter=n_iter)
        ptp_src.run()
        T_ns    = ptp_src.sync_period*1e9
    else:
        ptp_src = ptp.reader.Reader(args.file)
        ptp_src.process()
        T_ns    = 1e9/4

    # Iterate over the estimators
    estimators = [k for k in est_op.keys()] if (args.estimator == 'all') \
                  else [args.estimator]

    for estimator in estimators:
        # For the sample filters estimators that require the drift compensation
        # provided by ls, first we need to find the best window length for ls
        # and then run it.
        if (re.search("-ls$", estimator)):
            if (est_op["ls"]["N_best"] is None):
                max_te_vs_window(ptp_src.data, estimator="ls")

            ls = ptp.ls.Ls(est_op["ls"]["N_best"], ptp_src.data, T_ns)
            ls.process(impl="eff")

        # Run max_te_vs_window function
        max_te_vs_window(ptp_src.data, estimator=estimator)

if __name__ == "__main__":
    main()

