import argparse, logging, sys
import ptp.reader
import ptp.ls
import ptp.metrics
import ptp.pktselection
import ptp.kalman
import ptp.frequency
import ptp.window


def main():
    parser = argparse.ArgumentParser(description="PTP log reader test")
    parser.add_argument('-f', '--file',
                        default="log.json",
                        help='JSON log file.')
    parser.add_argument('--optimizer',
                        default=False,
                        action='store_true',
                        help='Whether or not to optimize window length')
    parser.add_argument('--use-secs',
                        default=False,
                        action='store_true',
                        help="Use secs that were actually captured " +
                        "(i.e. do not infer secs)")
    parser.add_argument('--no-pps',
                        default=False,
                        action='store_true',
                        help="Do not look for reference timestamps from PPS RTC")
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
                               no_pps=args.no_pps, reverse_ms=True)
    reader.run(args.num_iter)

    # Nominal message in nanoseconds
    T_ns = reader.metadata["sync_period"]*1e9

    # Optimize window length configuration
    if (args.optimizer):
        window_optimizer = ptp.window.Optimizer(reader.data, T_ns)
        window_optimizer.process('all')
        est_op    = window_optimizer.est_op
        N_ls      = est_op["ls"]["N_best"]             # LS
        N_movavg  = est_op["sample-average"]["N_best"] # Moving average
        N_median  = est_op["sample-median"]["N_best"]  # Sample-median
        N_min     = est_op["sample-min"]["N_best"]     # Sample-minimum
        N_min_ls  = est_op["sample-min-ls"]["N_best"]  # Sample-minimum with LS
        N_mode    = est_op["sample-mode"]["N_best"]    # Sample-minimum
        N_mode_ls = est_op["sample-mode-ls"]["N_best"] # Sample-mode with LS
        N_ewma    = est_op["ls"]["N_best"]             # EWMA window
    else:
        N_ls      = 105
        N_movavg  = 16
        N_median  = 16
        N_min     = 16
        N_min_ls  = 16
        N_mode    = 16
        N_mode_ls = 16
        N_ewma    = 16

    # Least-squares estimator
    ls = ptp.ls.Ls(N_ls, reader.data, T_ns)
    ls.process("eff")

    # Raw frequency estimations (differentiation of raw time offset measurements)
    freq_delta = 64
    freq_estimator = ptp.frequency.Estimator(reader.data, delta=freq_delta)
    freq_estimator.process()
    freq_estimator.set_truth()

    # Kalman
    # kalman = ptp.kalman.Kalman(reader.data, T_ns/1e9)
    kalman = ptp.kalman.Kalman(reader.data, T_ns/1e9,
                               trans_cov = [[1, 0], [0, 1e-2]],
                               obs_cov = [[1e4, 0], [0, 1e2]])
    kalman.process()

    # Moving average
    pkts = ptp.pktselection.PktSelection(N_movavg, reader.data)
    pkts.process("average", avg_impl="recursive")

    # Sample-median
    pkts.set_window_len(N_median)
    pkts.process("median")

    # Sample-minimum
    pkts.set_window_len(N_min)
    pkts.process("min")
    pkts.set_window_len(N_min_ls)
    pkts.process("min", ls_impl="eff")

    # Exponentially weighted moving average
    pkts.set_window_len(N_ewma)
    pkts.process("ewma")

    # Sample-mode
    pkts.set_window_len(N_mode)
    pkts.process("mode")
    pkts.set_window_len(N_mode_ls)
    pkts.process("mode", ls_impl="eff")

    # PTP analyser
    analyser = ptp.metrics.Analyser(reader.data)
    analyser.plot_toffset_vs_time()
    analyser.plot_foffset_vs_time()

    # When the reference timestamps are available
    if (not args.no_pps):
        analyser.plot_toffset_err_vs_time(show_raw = False)
        analyser.plot_delay_vs_time(save=True)
        analyser.plot_delay_hist(save=True, n_bins=20)
        analyser.plot_mtie(show_raw = False)
        analyser.plot_max_te(show_raw=False, window_len = 200)

if __name__ == "__main__":
    main()


