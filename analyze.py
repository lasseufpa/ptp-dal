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


def _run_drift_estimation(data, cache):
    """Run frequency offset and time offset drift estimations"""

    # Raw frequency estimations (mostly for visualization)
    freq_delta = 64
    freq_estimator = ptp.frequency.Estimator(data, delta=freq_delta)
    freq_estimator.set_truth(delta=freq_delta)
    freq_estimator.optimize_to_y()
    freq_estimator.process()

    # Time offset drift estimations through the PI control loop
    damping, loopbw = freq_estimator.optimize_loop(cache=cache)
    freq_estimator.loop(damping = damping, loopbw = loopbw)


def _run_window_optimizer(data, T_ns, dataset_file, metric, disable_plot,
                          en_fine, force, max_window, cache):
    """Run tuner of window lengths"""

    window_optimizer = ptp.window.Optimizer(data, T_ns, dataset_file)
    window_optimizer.process('all',
                             error_metric = metric,
                             plot = (not disable_plot),
                             fine_pass = en_fine,
                             force = force,
                             max_window = max_window,
                             cache=cache)
    window_optimizer.print_results()
    return window_optimizer.get_results()


def _run_kalman(data, T_ns):
    """Run Kalman Filtering"""

    kalman = ptp.kalman.Kalman(data, T_ns/1e9,
                               trans_cov = [[1, 0], [0, 1e-2]],
                               obs_cov = [[1e4, 0], [0, 1e2]])
    kalman.process()


def _run_ls(data, N_ls, T_ns):
    """Run Least-squares estimator"""

    ls = ptp.ls.Ls(N_ls, data, T_ns)
    ls.process("eff")


def _run_pre_bias_compensation(data):
    """Compensate the bias of two-way time offset measurements prior to
     post-processing

    NOTE: the raw time offset measurements are processed directly by some packet
    selection operators (sample-average and EWMA), as well as by LS, Kalman and
    the PI loop. Thus, with correction of the bias of "x_est" here it is
    expected that the referred estimators also produce unbiased results.

    """
    bias = ptp.bias.Bias(data)
    corr = {
        'avg' : bias.calc_true_asymmetry(operator="raw")
    }

    if ('avg' in corr):
        bias.compensate(corr=corr['avg'], toffset_key="x_est")
    else:
        logging.warning("Can't compensate average asymmetry")


def _run_post_bias_compensation(data):
    """Compensate bias of results produced by some packet selection operators"""

    bias = ptp.bias.Bias(data)
    corr = {
        'median' : bias.calc_true_asymmetry(operator="median"),
        'min'    : bias.calc_true_asymmetry(operator="min"),
        'max'    : bias.calc_true_asymmetry(operator="max"),
        'mode'   : bias.calc_true_asymmetry(operator="mode")
    }

    # Sample-median
    if ('median' in corr):
        bias.compensate(corr=corr['median'], toffset_key="x_pkts_median")
    else:
        logging.warning("Can't compensate asymmetry of median")

    # Sample-minimum
    if ('min' in corr):
        bias.compensate(corr=corr['min'], toffset_key="x_pkts_min")
    else:
        logging.warning("Can't compensate asymmetry of minimum")

    # Sample-maximum
    if ('max' in corr):
        bias.compensate(corr=corr['max'], toffset_key="x_pkts_max")
    else:
        logging.warning("Can't compensate asymmetry of maximum")

    # Sample-mode
    if ('mode' in corr):
        bias.compensate(corr=corr['mode'], toffset_key="x_pkts_mode")
    else:
        logging.warning("Can't compensate asymmetry of mode")


def _run_pktselection(data, window_len):
    """Run packet selection algorithms"""

    # Moving average
    pkts = ptp.pktselection.PktSelection(window_len['movavg'], data)
    pkts.process("avg-recursive")

    # Sample-median
    pkts.set_window_len(window_len['median'])
    pkts.process("median")

    # Sample-minimum
    pkts.set_window_len(window_len['min'])
    pkts.process("min")

    # Sample-maximum
    pkts.set_window_len(window_len['max'])
    pkts.process("max")

    # Exponentially weighted moving average
    pkts.set_window_len(window_len['ewma'])
    pkts.process("ewma")

    # Sample-mode
    pkts.set_window_len(window_len['mode'])
    pkts.process("mode")


def _run_analyzer(data, metadata, dataset_file, source, no_processing=False):
    """Analyze results"""

    analyser = ptp.metrics.Analyser(data, dataset_file)
    analyser.save_metadata(metadata, save=True)

    # Start with the analysis that does not require processing algorithms. That
    # is, the analysis based on info readily available in the dataset or
    # generated by the reader processing.
    analyser.check_seq_id_gaps()
    if (source == "testbed"):
        analyser.calc_expected_delays(metadata, save=True)
        analyser.plot_temperature()
        analyser.plot_occupancy()
        analyser.plot_pps_err()
        analyser.plot_pps_rtc_foffset_est()
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
    analyser.ptp_exchanges_per_sec(save=True)
    analyser.delay_asymmetry(save=True)

    if (no_processing):
        analyser.plot_toffset_vs_time(show_raw=False, show_best=False,
                                      show_ls=False, show_pkts=False,
                                      show_kf=False, show_loop=False,
                                      show_true=True)
        return

    analyser.plot_toffset_vs_time()
    analyser.plot_foffset_vs_time()
    analyser.plot_toffset_err_hist()
    analyser.plot_toffset_err_vs_time(show_raw = False)
    analyser.plot_foffset_err_hist()
    analyser.plot_foffset_err_vs_time()
    analyser.plot_toffset_drift_vs_time()
    analyser.plot_toffset_drift_hist()
    analyser.plot_mtie(show_raw = False)
    analyser.plot_max_te(show_raw=False, window_len = 1000)
    analyser.toffset_err_stats(save=True)
    analyser.foffset_err_stats(save=True)
    analyser.toffset_drift_err_stats(save=True)
    analyser.ranking(metric="max-te", save=True)
    analyser.ranking(metric="mtie", save=True)
    analyser.ranking(metric="rms", save=True)
    analyser.ranking(metric="std", save=True)


def main():
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
    parser.add_argument('--no-optimizer',
                        default=False,
                        action='store_true',
                        help='Whether or not to optimize window length')
    parser.add_argument('--no-optimizer-plots',
                        default=False,
                        action='store_true',
                        help='Whether to disable window optimizer plots')
    parser.add_argument('--no-cache',
                        default=False,
                        action='store_true',
                        help='Whether to disable caching of optimal \
                        configurations')
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
    parser.add_argument('--optimizer-max-window',
                        default=8192,
                        type=int,
                        help='Maximum window length that the window optimizer \
                        can return for any algorithm.')
    parser.add_argument('--infer-secs',
                        default=False,
                        action='store_true',
                        help="Infer seconds rather than using the seconds that \
                        were actually captured.")
    parser.add_argument('--bias',
                        choices=['pre', 'post', 'both', 'none'],
                        default='both',
                        help="Compensate the bias prior to any post-processing \
                        (pre), after post-processing (post), both pre and \
                        post post-processing (both) or disable it ('none').")
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations.')
    parser.add_argument('-t', '--time-interval',
                        default=None,
                        help='Specific time interval to observe given as \
                        \"start:end\" in hours.')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level")
    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # Download the dataset if not available
    ds_manager = ptp.datasets.Datasets()
    ds_path    = ds_manager.download(args.file)

    # Define cache handler
    cache = None if args.no_cache else ptp.cache.Cache(ds_path)

    # Detect the source of the dataset
    ds_source = "testbed" if args.file.split("-")[0] == "serial" else \
                "simulation"
    logging.info(f"Dataset acquired from {ds_source}")

    # Run the source of PTP data
    if (ds_source == "testbed"):
        src = ptp.reader.Reader(ds_path, infer_secs = args.infer_secs,
                                reverse_ms = True)
        src.run(args.num_iter)
    else:
        src = ptp.simulation.Simulation()
        src.load(ds_path)

    if (args.time_interval):
        src.trim(args.time_interval)

    if (args.analyze_only):
        _run_analyzer(src.data, src.metadata, ds_path, ds_source,
                      no_processing=True)
        return

    # Nominal message period in nanoseconds
    T_ns = src.metadata["sync_period"]*1e9

    _run_outlier_detection(src.data)

    if (args.bias == 'pre' or args.bias == 'both'):
        _run_pre_bias_compensation(src.data)

    _run_drift_estimation(src.data, cache=cache)

    if (args.no_optimizer):
        window_lengths = default_window_lengths
    else:
        window_lengths = _run_window_optimizer(src.data, T_ns, ds_path,
                                               args.optimizer_metric,
                                               args.no_optimizer_plots,
                                               args.optimizer_fine,
                                               args.optimizer_force,
                                               args.optimizer_max_window,
                                               cache)

    _run_ls(src.data, window_lengths['ls'], T_ns)

    _run_kalman(src.data, T_ns)

    _run_pktselection(src.data, window_lengths)

    if (args.bias == 'post' or args.bias == 'both'):
        _run_post_bias_compensation(src.data)

    _run_analyzer(src.data, src.metadata, ds_path, ds_source)


if __name__ == "__main__":
    main()


