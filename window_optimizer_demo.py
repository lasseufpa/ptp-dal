"""Analyse the estimators performance as a function of window length
"""
import argparse
import ptp.runner, ptp.reader, ptp.window
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import time, re

def main():
    # Available estimators
    est_op = ptp.window.est_op
    est_choices = [k for k in est_op] + ['all']

    parser = argparse.ArgumentParser(description="Max|TE| vs window")
    parser.add_argument('-e', '--estimator',
                        default='all',
                        help='Window-based estimator',
                        choices=est_choices)
    exc_group = parser.add_mutually_exclusive_group()
    exc_group.add_argument('-f', '--file',
                        default=None,
                        help='Serial capture file.')
    exc_group.add_argument('-N', '--num-iter',
                        default=2000,
                        type=int,
                        help='Number of iterations if running simulation.')
    args = parser.parse_args()

    # Options to save and plot
    plot = True
    save = False

    if (args.file is None):
        # Run PTP simulation
        ptp_src  = ptp.runner.Runner(n_iter=args.num_iter)
        ptp_src.run()
        T_ns     = ptp_src.sync_period*1e9

    else:
        # Run reader process
        ptp_src  = ptp.reader.Reader(args.file)
        ptp_src.process()

        # Get sync period from metadata
        if (hasattr(ptp_src, 'metadata') and
            'sync_period' in ptp_src.metadata and
            ptp_src.metadata['sync_period'] is not None):
            xT_ns = ptp_src.metadata['sync_period']*1e9
        else:
            # FIXME we should use at least a command-line variable
            T_ns = 1e9/4

    # Optimize window lengths (based on Max|TE|)
    window_optimizer = ptp.window.Optimizer(ptp_src.data, T_ns)
    window_optimizer.process(args.estimator, file=args.file, save=save,
                             plot=plot)

if __name__ == "__main__":
    main()
