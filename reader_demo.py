import argparse, logging, sys
import ptp.reader
import ptp.ls
import ptp.metrics
import ptp.kalman
import ptp.frequency


def main():
    parser = argparse.ArgumentParser(description="PTP log reader test")
    parser.add_argument('-f', '--file',
                        default="log.json",
                        help='JSON log file.')
    parser.add_argument('--infer-secs',
                        default=False,
                        action='store_true',
                        help="Infer timestamp secs from captured ns values")
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
    reader = ptp.reader.Reader(args.file)
    reader.process(args.num_iter, infer_secs=args.infer_secs,
                   no_pps=args.no_pps)

    # Least-squares estimator
    N    = 64                   # LS observation window length
    T_ns = 1e9/4                # Nominal message period in nanoseconds
    ls = ptp.ls.Ls(N, reader.data, T_ns)
    ls.process("eff")

    # Raw frequency estimations (differentiation of raw time offset measurements)
    freq_estimator = ptp.frequency.Estimator(reader.data)
    freq_estimator.process()

    # Kalman
    kalman = ptp.kalman.Kalman(reader.data, T_ns/1e9)
    kalman.process()

    # PTP analyser
    analyser = ptp.metrics.Analyser(reader.data)
    analyser.plot_toffset_vs_time(show_ls=True, show_true=False, show_kf=True,
                                  n_skip_kf=1000, save=True)
    analyser.plot_foffset_vs_time(show_ls=True, show_raw=False, show_kf=True,
                                  n_skip_kf=1000, show_true=False, save=True)

    # When the reference timestamps are available
    if (not args.no_pps):
        analyser.plot_toffset_err_vs_time(show_raw = True,
                                          show_ls = True,
                                          show_pkts = False,
                                          show_kf = True,
                                          save = True)
        analyser.plot_delay_vs_time(save=True)
        analyser.plot_delay_hist(save=True, n_bins=20)


if __name__ == "__main__":
    main()


