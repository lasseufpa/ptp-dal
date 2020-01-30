#!/usr/bin/env python

import argparse, logging, sys
import ptp.reader
import ptp.ls
import ptp.metrics
import ptp.pktselection
import ptp.kalman
import ptp.frequency
import ptp.window
import ptp.outlier
import ptp.bias


def main():
    parser = argparse.ArgumentParser(description="PTP log reader test")
    parser.add_argument('-f', '--file',
                        default="log.json",
                        help='JSON log file.')
    parser.add_argument('--no-optimizer',
                        default=False,
                        action='store_true',
                        help='Whether or not to optimize window length')
    parser.add_argument('--no-optimizer-plots',
                        default=False,
                        action='store_true',
                        help='Whether to disable window optimizer plots')
    parser.add_argument('--optimizer-fine',
                        default=False,
                        action='store_true',
                        help='Whether to enable window optimizer fine pass')
    parser.add_argument('--optimizer-force',
                        default=False,
                        action='store_true',
                        help='Force window optimizer processing even if \
                        already done previously')
    parser.add_argument('--optimizer-metric',
                        default='max-te',
                        help='Estimation error metric for window tuning',
                        choices=['max-te', 'mse'])
    parser.add_argument('--use-secs',
                        default=False,
                        action='store_true',
                        help="Use secs that were actually captured " +
                        "(i.e. do not infer secs)")
    parser.add_argument('--bias',
                        choices=['pre', 'post', 'both', 'none'],
                        default='both',
                        help="Compensate the bias prior to any post-processing \
                        (pre), after post-processing (post), both pre and \
                        post post-processing (both) or disable it ('none') \
                        (default: 'both')")
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations.')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level")
    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # Run PTP simulation
    reader = ptp.reader.Reader(args.file, infer_secs=(not args.use_secs),
                               reverse_ms=True)
    reader.run(args.num_iter)

    # Outlier detection
    outlier = ptp.outlier.Outlier(reader.data)
    outlier.process(c=2)

    # Nominal message in nanoseconds
    if (reader.metadata is not None and "sync_period" in reader.metadata):
        T_ns = reader.metadata["sync_period"]*1e9
    else:
        T_ns = 1e9/4

    # Compensate the bias of two-way time offset measurements prior to
    # post-processing
    #
    # NOTE the raw time offset measurements are processed directly by some
    # packet selection operators (sample-average and EWMA), as well as by LS and
    # Kalman. Thus, with correction of the bias of "x_est" here it is expected
    # that the referred estimators also produce unbiased results.
    if (args.bias == 'pre' or args.bias == 'both'):
        bias = ptp.bias.Bias(reader.data)
        corr = bias.calc_true_asymmetry(operator="raw")
        bias.compensate(corr=corr, toffset_key="x_est")

    # Raw frequency estimations (mostly for visualization)
    freq_delta = 64
    freq_estimator = ptp.frequency.Estimator(reader.data, delta=freq_delta)
    freq_estimator.set_truth(delta=freq_delta)
    freq_estimator.optimize_to_y()
    freq_estimator.process()

    # Time offset drift estimations through the PI control loop
    damping, loopbw = freq_estimator.optimize_loop()
    freq_estimator.loop(damping = damping, loopbw = loopbw)

    # Optimize window length configuration
    if (not args.no_optimizer):
        window_optimizer = ptp.window.Optimizer(reader.data, T_ns)
        window_optimizer.process('all',
                                 error_metric=args.optimizer_metric,
                                 file=args.file,
                                 plot=(not args.no_optimizer_plots),
                                 fine_pass=args.optimizer_fine,
                                 force=args.optimizer_force)
        window_optimizer.save()
        est_op    = window_optimizer.est_op
        N_ls      = est_op["ls"]["N_best"]             # LS
        N_movavg  = est_op["sample-average"]["N_best"] # Moving average
        N_median  = est_op["sample-median"]["N_best"]  # Sample-median
        N_min     = est_op["sample-min"]["N_best"]     # Sample-minimum
        N_max     = est_op["sample-max"]["N_best"]     # Sample-maximum
        N_mode    = est_op["sample-mode"]["N_best"]    # Sample-mode
        N_ewma    = est_op["sample-ewma"]["N_best"]    # EWMA

        print("Tuned window lengths:")
        for i in est_op:
            print("%20s: %d" %(i, est_op[i]["N_best"]))
    else:
        N_ls      = 105
        N_movavg  = 16
        N_median  = 16
        N_min     = 16
        N_max     = 16
        N_mode    = 16
        N_ewma    = 16

    # Least-squares estimator
    ls = ptp.ls.Ls(N_ls, reader.data, T_ns)
    ls.process("eff")

    # Kalman
    # kalman = ptp.kalman.Kalman(reader.data, T_ns/1e9)
    kalman = ptp.kalman.Kalman(reader.data, T_ns/1e9,
                               trans_cov = [[1, 0], [0, 1e-2]],
                               obs_cov = [[1e4, 0], [0, 1e2]])
    kalman.process()

    # Moving average
    pkts = ptp.pktselection.PktSelection(N_movavg, reader.data)
    pkts.process("avg-recursive")

    # Sample-median
    pkts.set_window_len(N_median)
    pkts.process("median")

    # Sample-minimum
    pkts.set_window_len(N_min)
    pkts.process("min")

    # Sample-maximum
    pkts.set_window_len(N_max)
    pkts.process("max")

    # Exponentially weighted moving average
    pkts.set_window_len(N_ewma)
    pkts.process("ewma")

    # Sample-mode
    pkts.set_window_len(N_mode)
    pkts.process("mode")

    # Compensate bias of results produced by some packet selection operators
    if (args.bias == 'post' or args.bias == 'both'):
        bias = ptp.bias.Bias(reader.data)

        # Sample-median
        corr_median = bias.calc_true_asymmetry(operator="median")
        bias.compensate(corr=corr_median, toffset_key="x_pkts_median")

        # Sample-minimum
        corr_min = bias.calc_true_asymmetry(operator="min")
        bias.compensate(corr=corr_min, toffset_key="x_pkts_min")

        # Sample-maximum
        corr_max = bias.calc_true_asymmetry(operator="max")
        bias.compensate(corr=corr_max, toffset_key="x_pkts_max")

        # Sample-mode
        corr_mode = bias.calc_true_asymmetry(operator="mode")
        bias.compensate(corr=corr_mode, toffset_key="x_pkts_mode")

    # PTP analyser
    analyser = ptp.metrics.Analyser(reader.data, args.file)
    analyser.save_metadata(reader.metadata)
    analyser.check_seq_id_gaps()
    analyser.plot_toffset_vs_time()
    analyser.plot_foffset_vs_time()
    analyser.plot_temperature()
    analyser.plot_occupancy()
    analyser.plot_pps_err()
    analyser.plot_toffset_err_hist()
    analyser.plot_toffset_err_vs_time(show_raw = False)
    analyser.plot_foffset_err_hist()
    analyser.plot_foffset_err_vs_time()
    analyser.plot_delay_vs_time()
    analyser.plot_delay_vs_time(split=True)
    analyser.plot_delay_hist(n_bins=50)
    analyser.plot_delay_hist(split=True, n_bins=50)
    analyser.plot_delay_est_err_vs_time()
    analyser.plot_delay_asym_hist(n_bins=50)
    analyser.plot_delay_asym_vs_time()
    analyser.plot_pdv_vs_time()
    analyser.plot_pdv_hist()
    analyser.plot_ptp_exchange_interval_vs_time()
    analyser.plot_toffset_drift_vs_time()
    analyser.plot_toffset_drift_hist()
    analyser.plot_mtie(show_raw = False)
    analyser.plot_max_te(show_raw=False, window_len = 1000)
    analyser.ptp_exchanges_per_sec(save=True)
    analyser.delay_asymmetry(save=True)
    analyser.toffset_err_stats(save=True)
    analyser.foffset_err_stats(save=True)
    analyser.toffset_drift_err_stats(save=True)


if __name__ == "__main__":
    main()


