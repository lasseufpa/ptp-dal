#!/usr/bin/env python

"""Analyse the estimators performance as a function of window length
"""
import argparse, logging, sys
import ptp.simulation
import ptp.reader
import ptp.window
import ptp.frequency
import ptp.cache
import ptp.datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import time, re

def main():
    # Available estimators
    est_op = ptp.window.Optimizer.est_op
    est_choices = [k for k in est_op] + ['all']

    parser = argparse.ArgumentParser(description="Window length optimizer")
    parser.add_argument('-e', '--estimator',
                        default='all',
                        help='Window-based estimator',
                        choices=est_choices)
    parser.add_argument('-m', '--metric',
                        default='max-te',
                        help='Estimation error metric for performance assessment',
                        choices=['max-te', 'mse'])
    parser.add_argument('-p', '--plot',
                        default=False,
                        action='store_true',
                        help='Whether or not to plot results')
    parser.add_argument('--save-plot',
                        default=False,
                        action='store_true',
                        help='Whether or not to save plot results')
    parser.add_argument('--global-plot',
                        default=False,
                        action='store_true',
                        help='Whether or not to plot global curve')
    parser.add_argument('--no-plot-info',
                        default=False,
                        action='store_true',
                        help='Whether or not to save window information in plot')
    parser.add_argument('-s', '--save',
                        default=False,
                        action='store_true',
                        help='Whether or not to save window configurations')
    parser.add_argument('-f', '--file',
                        default=None,
                        help='Serial capture file.')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations')
    parser.add_argument('--no-stop',
                        default=False,
                        action='store_true',
                        help='Do not apply early stopping')
    parser.add_argument('--force',
                        default=False,
                        action='store_true',
                        help='Force processing even if already done previously')
    parser.add_argument('--fine',
                        default=False,
                        action='store_true',
                        help='Whether to enable window optimizer fine pass')
    parser.add_argument('--no-cache',
                        default=False,
                        action='store_true',
                        help='Whether to disable save optimal configuration \
                        as cache.')
    parser.add_argument('--max-window',
                        default=8192,
                        type=int,
                        help='Maximum window length that the window optimizer \
                        can return for any algorithm.')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level")
    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # Download the dataset if not available
    ds_manager = ptp.datasets.Datasets()
    ds_path    = ds_manager.download(args.file)

    # Load the data
    if (args.file.split("-")[0] == "serial"):
        # Testbed
        src = ptp.reader.Reader(ds_path)
        src.run()
    else:
        # Simulation
        src = ptp.simulation.Simulation()
        src.load(ds_path)

    # Get sync period from metadata
    if (hasattr(src, 'metadata') and
        'sync_period' in src.metadata and
        src.metadata['sync_period'] is not None):
        T_ns = src.metadata['sync_period']*1e9
    else:
        # FIXME we should use at least a command-line variable
        T_ns = 1e9/4

    # Define cache object
    cache = None if args.no_cache else ptp.cache.Cache(args.file)

    # Time offset drift estimations through the PI control loop
    freq_estimator  = ptp.frequency.Estimator(src.data)
    damping, loopbw = freq_estimator.optimize_loop(cache=cache)
    freq_estimator.loop(damping = damping, loopbw = loopbw)

    # Optimize window lengths
    window_optimizer = ptp.window.Optimizer(src.data, T_ns, args.file)
    window_optimizer.process(args.estimator, error_metric=args.metric,
                             cache=cache, plot=args.plot,
                             early_stopping=(not args.no_stop),
                             save_plot=args.save_plot, force=args.force,
                             plot_info=(not args.no_plot_info),
                             global_plot=args.global_plot, fine_pass=args.fine,
                             max_window=args.max_window)

if __name__ == "__main__":
    main()
