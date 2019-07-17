"""Analyse the estimators performance as a function of window length
"""
import argparse
import ptp.runner, ptp.reader, ptp.max_te_vs_window
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import time, re

def main():
    # Available estimators
    est_op = ptp.max_te_vs_window.est_op
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

    # Options to save and plot
    plot = True
    save = False
    file = None if args.file is None else args.file

    if (args.file is None):
        # Run PTP simulation
        n_iter   = 2000
        ptp_src  = ptp.runner.Runner(n_iter=n_iter)
        ptp_src.run()
        T_ns     = ptp_src.sync_period*1e9

    else:
        # Run reader process
        ptp_src  = ptp.reader.Reader(args.file)
        ptp_src.process()
        T_ns     = 1e9/4

    # Run Max|TE| vs window
    max_te_vs_w = ptp.max_te_vs_window.MaxTeVsWindow(ptp_src.data, T_ns)
    max_te_vs_w.process(args.estimator, file=file, save=save, plot=plot)

if __name__ == "__main__":
    main()