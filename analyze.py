#!/usr/bin/env python

import logging, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ptp.reader
import ptp.ls
import ptp.metrics
import ptp.pktselection
import ptp.kalman
import ptp.frequency
import ptp.window
import ptp.outlier
import ptp.bias
import ptp.datasets
import ptp.cache
import ptp.simulation


default_window_lengths = {
    'ls'      : 105,
    'movavg'  : 16,
    'median'  : 16,
    'min'     : 16,
    'max'     : 16,
    'mode'    : 16,
    'ewma'    : 16
}


def _run_outlier_detection(data):
    """Run Outlier detection"""

    outlier = ptp.outlier.Outlier(data)
    outlier.process(c=2)


def _calc_max_drift_est_transient(skip, drift_comp, optimizer_max_window,
                                  n_data):
    """Compute the maximum allowed transient for drift estimations

    The window-based algorithms produce estimates by the end of the observation
    windows, such that the first estimate is produced by the end of the first
    window. Similarly, the TLL and Kalman filter produce estimates after the
    their transients. To ensure that we have estimates throughout the portion of
    the dataset that is analyzed (after skipping samples), we must limit the
    window lengths and transient phases accordingly. This limitation is imposed
    in terms of the maximum fraction of the dataset that a transient phase can
    occupy.

    With drift compensation on window-based algorithms, the first window can
    only be computed once drift estimates become available in the
    dataset. Hence, in this case, the two transients are added together (they
    are consecutive). Otherwise, the two transients occur in parallel. In any
    case, the value provided through command-line argument "--skip" controls the
    maximum total transient as a fraction of the dataset length.

    """

    window_transient = optimizer_max_window / n_data
    assert(window_transient < skip), \
        "Max window set for the optimizer exceeds the max acceptable transient"

    if (drift_comp):
        max_drift_transient = skip - window_transient
    else:
        max_drift_transient = skip

    if (drift_comp and max_drift_transient < 0.5*window_transient):
        # Ideally, the maximum window used for packet selection/filtering still
        # leaves sufficient room for the drift estimation transient. Otherwise,
        # the drift estimation performance may deteriorate. For example, when
        # estimating drifts based on the TLL, sometimes the best drift estimates
        # come from TLL configurations with relatively long transients.
        logging.warning("Drift estimator's transient is less than half of the "
                        "maximum window transient")

    return max_drift_transient


def _run_foffset_estimation(data, N=64, strategy="two-way", loss="mse",
                            max_transient=0.2, truth_only=False):
    """Run frequency offset estimations"""
    freq_estimator = ptp.frequency.Estimator(data, delta=N)
    freq_estimator.set_truth()

    if (not truth_only):
        freq_estimator.optimize_to_y(strategy, loss=loss,
                                     max_window_span=max_transient)
        freq_estimator.process(strategy)


def _run_drift_estimation(data, strategy, pkts=False, loss="max-error",
                          cache=None, cache_id='drift_est', force=False,
                          max_transient=0.2):
    """Run time offset drift estimations through the PI control loop"""
    assert(strategy in ["loop", "unbiased-two-way", "unbiased-one-way",
                        "unbiased-one-way-reversed"])

    freq_estimator  = ptp.frequency.Estimator(data, pkts=pkts)

    if (strategy == "loop"):
        damping, loopbw = freq_estimator.optimize_loop(
            loss=loss,
            cache=cache,
            cache_id=cache_id + "_loop",
            force=force,
            max_transient=max_transient)
        freq_estimator.loop(damping = damping, loopbw = loopbw,
                            settling=max_transient)
    else:
        # Select the unbiased frequency offset computation strategy
        strategy = strategy.replace("unbiased-", "")
        freq_estimator.optimize_to_drift(strategy,
                                         loss=loss,
                                         max_window_span=max_transient,
                                         cache=cache,
                                         cache_id=cache_id,
                                         force=force)
        freq_estimator.process(strategy)
        freq_estimator.estimate_drift()


def _run_window_optimizer(data, disable_list, T_ns, metric, en_fine, force,
                          max_window, early_stopping, cache, drift_comp, bias,
                          bias_est, batch_size):
    """Run tuner of window lengths"""

    # Options used by the algorithms executed internally within the optimizer
    algo_opts = {
        'drift_comp' : drift_comp, # whether to apply drift compensation
        'bias_corr_mode' : bias,   # bias correction mode
        'bias_est' : bias_est,     # bias estimates
        'batch_size' : batch_size  # batch size
    }

    window_optimizer = ptp.window.Optimizer(data, T_ns, algo_opts)

    # Window-based estimators to optimize
    estimator_list = list(window_optimizer.est_op.keys())

    # Remove selected estimators
    for estimator in disable_list:
        if (estimator in estimator_list):
            estimator_list.remove(estimator)

    window_optimizer.process(estimator_list,
                             error_metric = metric,
                             fine_pass = en_fine,
                             force = force,
                             max_window = max_window,
                             early_stopping = early_stopping,
                             cache=cache)
    window_optimizer.print_results()
    return window_optimizer.get_results()


def _run_kalman(data, T_ns, cache, force):
    """Run Kalman Filtering"""

    kf = ptp.kalman.KalmanFilter(data, T_ns/1e9)
    kf.optimize(error_metric='mse', cache=cache, force=force)
    kf.process()


def _run_ls(data, N_ls, T_ns, batch_size):
    """Run Least-squares estimator"""

    ls = ptp.ls.Ls(N_ls, data, T_ns)
    ls.process("eff", batch_size=batch_size)


def _compute_ideal_bias_estimates(data):
    """Compute the ideal bias estimates

    Args:
        data : Dataset.

    Returns:
        Dictionary with the ideal bias estimates.

    """
    bias = ptp.bias.Bias(data)

    bias_est = {}
    for metric in ['avg', 'min', 'max', 'median', 'mode']:
        bias_est[metric] = bias.calc_true_asymmetry(metric=metric)

    return bias_est


def _run_pre_bias_compensation(data, corr):
    """Compensate the bias of two-way time offset measurements before processing

    The raw time offset measurements are processed directly by some packet
    selection operators (sample-average and EWMA), as well as by LS, Kalman and
    the PI loop. Thus, with correction of the bias of "x_est" here, it is
    expected that the referred estimators also produce unbiased results.

    Arguments:
        data : Dataset.
        corr : Dictionary with bias corrections to apply.

    """
    bias = ptp.bias.Bias(data)

    if ('avg' in corr and corr['avg']):
        bias.compensate(corr=corr['avg'], toffset_key="x_est")
    else:
        logging.warning("Can't compensate average asymmetry")


def _run_post_bias_compensation(data, corr):
    """Compensate bias of results produced by some packet selection operators

    These operators process timestamp differences (t21 and t43) instead of the
    raw time offset measurements "x_est". Furthermore, the packet selection
    operators experience distinct asymmetries (max/min/mode/median). If the goal
    was to pre-compensate their biases before running the algorithms, it would
    be necessary to maintain different pre-compensated copies of t21 and t43,
    each corrected by the specific asymmetry of interest. To avoid this
    overhead, it's convenient to correct the estimates resulting from each
    algorithm instead. This function corrects such estimates in place.

    Args:
        data : Dataset.
        corr : Dictionary with bias corrections to apply.

    """

    bias = ptp.bias.Bias(data)

    for metric in ['median', 'min', 'max', 'mode']:
        if (metric in corr and corr[metric]):
            bias.compensate(corr=corr[metric],
                            toffset_key=f"x_pkts_{metric}")
        else:
            logging.warning(f"Can't compensate asymmetry of {metric}")


def _run_pktselection(data, window_len, batch_size, drift_comp=True,
                      sample_avg=True, sample_median=True, sample_min=True,
                      sample_max=True, sample_mode=True, ewma=True):
    """Run packet selection algorithms"""

    pkts = ptp.pktselection.PktSelection(N=None, data=data)

    # Sample-average
    if (sample_avg):
        pkts.set_window_len(window_len['movavg'])
        pkts.process("avg", drift_comp=drift_comp, batch_size=batch_size)

    # Sample-median
    if (sample_median):
        pkts.set_window_len(window_len['median'])
        pkts.process("median", drift_comp=drift_comp, batch_size=batch_size)

    # Sample-minimum
    if (sample_min):
        pkts.set_window_len(window_len['min'])
        pkts.process("min", drift_comp=drift_comp, batch_size=batch_size)

    # Sample-maximum
    if (sample_max):
        pkts.set_window_len(window_len['max'])
        pkts.process("max", drift_comp=drift_comp, batch_size=batch_size)

    # Exponentially weighted moving average
    if (ewma):
        pkts.set_window_len(window_len['ewma'])
        pkts.process("ewma", drift_comp=drift_comp, batch_size=batch_size)

    # Sample-mode
    if (sample_mode):
        pkts.set_window_len(window_len['mode'])
        pkts.process("mode", drift_comp=drift_comp, batch_size=batch_size)


def _run_analyzer(data, metadata, dataset_file, source, eps_format, dpi,
                  uselatex, skip, prefix=None, cache=None, save=True,
                  no_processing=False):
    """Analyze results"""

    save_format = 'eps' if eps_format else 'png'

    analyser = ptp.metrics.Analyser(data, dataset_file, prefix=prefix,
                                    usetex=uselatex, save_format=save_format,
                                    dpi=dpi, cache=cache, skip=skip)

    analyser.save_metadata(metadata, save=save)

    # Nominal message period in seconds
    T = metadata["sync_period"]

    # Start with the analysis that does not require processing algorithms. That
    # is, the analysis based on info readily available in the dataset or
    # generated by the reader processing.
    analyser.check_seq_id_gaps()
    if (source == "testbed"):
        analyser.calc_expected_delays(metadata, save=save)
        analyser.plot_temperature(save=save)
        analyser.plot_occupancy(save=save)
        analyser.plot_pps_err(save=save)
        analyser.plot_pps_rtc_foffset_est(save=save)
    analyser.plot_delay_vs_time(save=save)
    analyser.plot_delay_vs_time(split=True, save=save)
    analyser.plot_delay_hist(n_bins=50, save=save)
    analyser.plot_delay_hist(n_bins=50, show_raw=False, save=save)
    analyser.plot_delay_hist(split=True, n_bins=50, save=save)
    analyser.plot_delay_est_err_vs_time(save=save)
    analyser.plot_delay_asym_hist(n_bins=50, save=save)
    analyser.plot_delay_asym_vs_time(save=save)
    analyser.plot_pdv_vs_time(save=save)
    analyser.plot_pdv_hist(save=save)
    analyser.plot_ptp_exchange_interval_vs_time(save=save)
    analyser.ptp_exchanges_per_sec(save=save)
    analyser.delay_asymmetry(save=save)

    if (no_processing):
        analyser.plot_toffset_vs_time(show_raw=False, show_best=False,
                                      show_ls=False, show_pkts=False,
                                      show_kf=False, show_loop=False,
                                      show_true=True, save=save)
        return

    analyser.plot_toffset_vs_time(save=save)
    analyser.plot_foffset_vs_time(save=save)
    analyser.plot_toffset_err_hist(save=save)
    analyser.plot_toffset_err_vs_time(show_raw = False, save=save)
    analyser.plot_foffset_err_hist(save=save)
    analyser.plot_foffset_err_vs_time(save=save)
    analyser.plot_toffset_drift_vs_time(save=save)
    analyser.plot_toffset_drift_hist(save=save)
    analyser.plot_mtie(period = T, save=save)
    analyser.plot_mtie(period = T, show_raw = False, save=save)
    analyser.plot_max_te(window_len = int(60/T), save=save)
    analyser.plot_max_te(show_raw=False, window_len = int(60/T), save=save)
    analyser.plot_error_vs_window(save=save)
    analyser.window_optimizer_results(save=save)
    analyser.toffset_err_stats(save=save)
    analyser.foffset_err_stats(save=save)
    analyser.toffset_drift_err_stats(save=save)
    analyser.rank_algorithms(metric="max-te", max_te_win_len = int(60/T),
                             save=save)
    analyser.rank_algorithms(metric="mtie", save=save)
    analyser.rank_algorithms(metric="rms", save=save)
    analyser.rank_algorithms(metric="std", save=save)
    analyser.save_maxte_and_mtie_cache()


def parse_args():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="PTP Analyser",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file',
                        default="log.json",
                        help='JSON dataset file.')
    parser.add_argument('--analyze-only',
                        default=False,
                        action='store_true',
                        help="Run analyzer plots only and don't run any \
                        post-processing. The analyzer will then process only \
                        the data that is already available in the dataset and \
                        the data produced by the dataset reader, such as  \
                        two-way time offset and delay measurements. Useful \
                        to inspect results without much wait.")
    parser.add_argument('--disable',
                        choices=["ewma", "sample-average", "sample-min",
                                 "sample-max", "sample-mode", "sample-median",
                                 "ls", "kalman"],
                        default=None,
                        nargs='+',
                        help="Algorithms to disable")
    parser.add_argument('--no-optimizer',
                        default=False,
                        action='store_true',
                        help='Whether or not to optimize window length')
    parser.add_argument('--no-cache',
                        default=False,
                        action='store_true',
                        help='Whether to disable caching of optimal \
                        configurations')
    parser.add_argument('--infer-secs',
                        default=False,
                        action='store_true',
                        help="Infer seconds rather than using the seconds that \
                        were actually captured.")
    parser.add_argument('--batch-size',
                        default=4096,
                        type=int,
                        help='Maximum number of observation windows processed \
                        at once on window-based algorithms.')
    parser.add_argument('--skip',
                        default=0.2,
                        type=float,
                        help='Fraction of the dataset to skip on analysis to \
                        ignore transient effects.')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations.')
    parser.add_argument('-t', '--time-interval',
                        default=None,
                        help='Specific time interval to observe given as \
                        \"start:end\" in hours.')
    parser.add_argument('--prefix',
                        default=None,
                        help='Prefix to prepend to saved plot and cache files.')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level")

    p_opts = parser.add_argument_group('Packet Selection/Filtering Options')
    p_opts.add_argument('--pkts-no-drift-comp',
                        default=False,
                        action='store_true',
                        help='Whether to disable the drift compensation step '
                        'applied within packet selection algorithms.')

    b_opts = parser.add_argument_group('Bias Correction Options')
    b_opts.add_argument('--bias',
                        choices=['pre', 'post', 'both', 'none'],
                        default='both',
                        help="Compensate the bias prior to any post-processing \
                        (pre), after post-processing (post), both pre and \
                        post post-processing (both) or disable it ('none').")

    o_opts = parser.add_argument_group('Window Optimizer Options')
    o_opts.add_argument('--optimizer-fine',
                        default=False,
                        action='store_true',
                        help='Whether to enable window optimizer fine pass')
    o_opts.add_argument('--optimizer-force',
                        default=False,
                        action='store_true',
                        help='Force window optimizer processing even if \
                        already done previously')
    o_opts.add_argument('--optimizer-metric',
                        default='max-te',
                        help='Estimation error metric for window tuning',
                        choices=['max-te', 'mse'])
    o_opts.add_argument('--optimizer-max-window',
                        default=8192,
                        type=int,
                        help='Maximum window length that the window optimizer \
                        can return for any algorithm.')
    o_opts.add_argument('--optimizer-no-stop',
                        default=False,
                        action='store_true',
                        help='Do not apply early stopping on window optimizer')

    d_opts = parser.add_argument_group('Drift Estimation Options')
    d_opts.add_argument('--drift-est-strategy',
                        default="loop",
                        choices=["loop", "unbiased-two-way", "unbiased-one-way",
                                 "unbiased-one-way-reversed"],
                        help='Drift estimation strategy. Select \"loop\" to \
                        use the drift estimates produced by the TLL PI loop or \
                        \"unbiased\" to use the conventional unbiased \
                        frequency offset estimator based on intervals measured \
                        at the slave and the master. Select specifically which \
                        unbiased estimation formulation to use: two-way, \
                        one-way, or reversed one-way.')
    d_opts.add_argument('--drift-est-pkts',
                        choices=["sample-min", "sample-max", None],
                        default=None,
                        help='Apply packet selection pre-filtering before \
                        frequency offset and drift estimations. Valid when \
                        using the unbiased frequency offset estimation \
                        strategy.')
    d_opts.add_argument('--drift-est-loss',
                        default="max-error",
                        choices=["max-error", "mse"],
                        help='Loss function used to optimize the drift \
                        estimator.')

    plot_opts = parser.add_argument_group('Plot Options')
    plot_opts.add_argument('--eps',
                           default=False,
                           action='store_true',
                           help='Whether to save images in .eps format.')
    plot_opts.add_argument('--dpi',
                           type=int,
                           default=300,
                           help='Images resolution in dots per inch.')
    plot_opts.add_argument('--latex',
                           default=False,
                           action='store_true',
                           help='Render plots using LaTeX.')

    cal_opts = parser.add_argument_group('Delay Asymmetry Calibration Options')
    cal_opts.add_argument('--cal-quantum',
                          default=50,
                          type=int,
                          help="Quantization step.")
    cal_opts.add_argument('--cal-decimation',
                          default=1,
                          type=int,
                          help="Decimation ratio. A value of 1 means decimation "
                          "is disabled.")
    cal_opts.add_argument('--cal-drift-comp',
                          default=False,
                          action='store_true',
                          help='Whether to apply drift compensation on '
                          'the delta_d estimates used for calibration. Useful '
                          'when the drift affecting delta_d is expected to be '
                          'larger than the drift estimation error.')
    cal_opts.add_argument('--cal-prob-thresh',
                          default=[0.001, 0.001],
                          nargs=2,
                          type=float,
                          help="Probability thresholds.")
    cal_opts.add_argument('--cal-static-thresh',
                          default=False,
                          action='store_true',
                          help='Whether to disable automatic threshold tuning.')
    return parser.parse_args()


def setup(args):
    """Initial setup"""
    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)


def read_dataset(args):
    """Read the dataset"""
    ds = {}
    # Download the dataset if not available
    ds_manager = ptp.datasets.Datasets()
    ds['path'] = ds_manager.download(args.file)

    # Define cache handler
    ds['cache'] = None if args.no_cache else ptp.cache.Cache(ds['path'],
                                                             args.prefix)

    # Detect the source of the dataset
    ds['name']   = ds['path'].split("/")[-1]
    ds['source'] = "testbed" if ds['name'].split("-")[0] == "serial" else \
                   "simulation"
    logging.info(f"Dataset acquired from {ds['source']}")

    # Run the source of PTP data
    if (ds['source'] == "testbed"):
        src = ptp.reader.Reader(ds['path'], infer_secs = args.infer_secs,
                                reverse_ms = True)
        src.run(args.num_iter)
    else:
        src = ptp.simulation.Simulation()
        src.load(ds['path'])

    if (args.time_interval):
        src.trim(args.time_interval)

    ds['data'] = src

    return ds


def process(ds, args, kalman=True, ls=True, pktselection=True,
            detect_outliers=True):
    """Run the processing stages"""

    # Nominal message period in nanoseconds
    T_ns = ds['data'].metadata["sync_period"]*1e9

    # Message rate
    ptp_rate = 1 / ds['data'].metadata["sync_period"]

    # Drift compensation applied on packet selection algorithms during the main
    # processing and the window optimization processing
    drift_comp = not args.pkts_no_drift_comp

    if (detect_outliers):
        _run_outlier_detection(ds['data'].data)

    # Compute the maximum fraction of the dataset to be occupied by the drift
    # estimator's transient phase
    max_drift_transient = _calc_max_drift_est_transient(
        args.skip, drift_comp, args.optimizer_max_window, len(ds['data'].data))

    # Bias estimates
    bias_est = _compute_ideal_bias_estimates(ds['data'].data)
    if (args.bias == 'pre' or args.bias == 'both'):
        _run_pre_bias_compensation(ds['data'].data, bias_est)
        # NOTE: this pre-compensation stage affects the window optimization. In
        # contrast, the post-compensation stage below does not. Hence, the
        # window optimizer includes its post-compensation stage internally.

    # Estimate the frequency offset throughout the acquisition
    #
    # NOTE:
    #
    # 1) The truth values are accurate within +-8ns, so the difference between
    # two timestamps can contain an error within +-16ns. Use a window of at
    # least 32 seconds so that this error falls into the sub-ppb region when
    # divided by the interval. That is, (+-16 ns)/(32 sec) = +-1 ns/sec (ppb).
    #
    # 2) When the drift estimation executed by "_run_drift_estimation()" uses
    # unbiased frequency offset estimates, it also estimates the frequency
    # offset. With that, it overwrites any estimation produced by
    # "_run_foffset_estimation()". In this case, it is useless to estimate the
    # frequency offset here. Nevertheless, call the function regardless because
    # it will at least compute the "true frequency offset", which is useful for
    # visualization on plots (i.e., use argument "truth_only").
    _run_foffset_estimation(ds['data'].data,
                            N=int(ptp_rate * 32),
                            loss=args.drift_est_loss,
                            max_transient=max_drift_transient,
                            truth_only=(args.drift_est_strategy != "loop"))

    # Estimate the time offset drifts used for drift compensation on packet
    # selection algorithms.
    _run_drift_estimation(ds['data'].data, args.drift_est_strategy,
                          pkts=args.drift_est_pkts,
                          loss=args.drift_est_loss, cache=ds['cache'],
                          force=args.optimizer_force,
                          max_transient=max_drift_transient)

    if (args.no_optimizer):
        window_lengths = default_window_lengths
    else:
        window_lengths = _run_window_optimizer(ds['data'].data, args.disable,
                                               T_ns, args.optimizer_metric,
                                               args.optimizer_fine,
                                               args.optimizer_force,
                                               args.optimizer_max_window,
                                               (not args.optimizer_no_stop),
                                               ds['cache'],
                                               drift_comp,
                                               args.bias,
                                               bias_est,
                                               args.batch_size)

    if ("ls" not in args.disable):
        _run_ls(ds['data'].data, window_lengths['ls'], T_ns, args.batch_size)

    if ("kalman" not in args.disable):
        _run_kalman(ds['data'].data, T_ns, cache=ds['cache'],
                    force=args.optimizer_force)

    if (pktselection):
        _run_pktselection(ds['data'].data, window_lengths, args.batch_size,
                          drift_comp = drift_comp,
                          sample_avg = ("sample-average" not in args.disable),
                          sample_median = ("sample-median" not in args.disable),
                          sample_min = ("sample-min" not in args.disable),
                          sample_max = ("sample-max" not in args.disable),
                          sample_mode = ("sample-mode" not in args.disable),
                          ewma = ("ewma" not in args.disable))

    if (args.bias == 'post' or args.bias == 'both'):
        _run_post_bias_compensation(ds['data'].data, bias_est)


def analyze(ds, args, no_processing=False, save=True):
    """Analyze results"""
    _run_analyzer(ds['data'].data, ds['data'].metadata, ds['path'],
                  ds['source'], eps_format=args.eps, dpi=args.dpi,
                  uselatex=args.latex, skip=args.skip, prefix=args.prefix,
                  cache=ds['cache'], save=save, no_processing=no_processing)


def main():
    args = parse_args()

    setup(args)

    ds = read_dataset(args)

    if (args.analyze_only):
        analyze(ds, args, no_processing=True)
        return

    process(ds, args)

    analyze(ds, args)


if __name__ == "__main__":
    main()


