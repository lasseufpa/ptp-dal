"""PTP metrics
"""
import math, logging, re, os, json, itertools
import ptp.cache
from datetime import timedelta
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)


NS_PER_MIN = (60 * 1e9)
est_keys = {"raw"                : {"label": "Raw Measurements",
                                    "marker": None,
                                    "show": True},
            "true"               : {"label": "True Values",
                                    "marker": None,
                                    "show": True},
            "pkts_avg_recursive" : {"label": "Sample-average (recursive)",
                                    "marker": "v",
                                    "show": True},
            "pkts_avg_normal"    : {"label": "Sample-average (normal)",
                                    "marker": "v",
                                    "show": True},
            "pkts_ewma"          : {"label": "EWMA",
                                    "marker": "v",
                                    "show": True},
            "pkts_median"        : {"label": "Sample-median",
                                    "marker": "v",
                                    "show": True},
            "pkts_min"           : {"label": "Sample-min",
                                    "marker": "v",
                                    "show": True},
            "pkts_max"           : {"label": "Sample-max",
                                    "marker": "v",
                                    "show": True},
            "pkts_mode"          : {"label": "Sample-mode",
                                    "marker": "v",
                                    "show": True},
            "ls_t2"              : {"label": "LSE (t2)",
                                    "marker": "x",
                                    "show": True},
            "ls_t1"              : {"label": "LSE (t1)",
                                    "marker": "x",
                                    "show": True},
            "ls_eff"             : {"label": "LS",
                                    "marker": "x",
                                    "show": True},
            "kf"                 : {"label": "KF",
                                    "marker": "d",
                                    "show": True},
            "loop"               : {"label": "PI Loop",
                                    "marker": "v",
                                    "show": True}}
g_markers = (".", "+", "o", "*", "v", "^", "<", ">", "8", "h",
             "s", "p", "d")


class Analyser():
    def __init__(self, data, file=None, prefix="", usetex=False,
                 save_format='png', dpi=300, cache=None):
        """PTP metrics analyser

        Args:
            data        : Array of objects with simulation data
            file        : Path of the file
            prefix      : Prefix to include on filenames when saving
            usetex      : Whether to use latex interpreter
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        self.data        = data
        self.path        = self._set_path(file)
        self.info        = os.path.join(self.path, 'info.txt')
        self.prefix      = prefix
        self.save_format = save_format
        self.dpi         = dpi
        self.usetex      = usetex
        self.cache       = cache

        if (cache is not None):
            assert(isinstance(cache, ptp.cache.Cache)), "Invalid cache object"

        if (usetex):
            aspect_ratio = 1.4
            max_width    = 3.39
            params = {
                'text.usetex'    : True,
                'axes.labelsize' : 8,
                'axes.titlesize' : 8,
                'font.size'      : 8,
                'legend.fontsize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'grid.color'     : 'k',
                'grid.linestyle' : ':',
                'grid.linewidth' : 0.5
            }
            matplotlib.rcParams.update(params)
        else:
            rc_figsize   = matplotlib.rcParams["figure.figsize"]
            aspect_ratio = rc_figsize[0]/rc_figsize[1]
            max_width    = rc_figsize[0]

        self.figsize  = (max_width, max_width/aspect_ratio)

        # Save some metrics results
        self.results  = {
            "max_te" : {},
            "mtie" : {}
        }


    def _set_path(self, file):
        """Define path to save plots

        Create a folder with the name of the file used to generate the metrics
        and save all the plots inside it, or if no file is used just save
        the metrics within the folder 'plots/'.

        Args:
            file : Path of the file

        """
        if (file):
            basename = os.path.splitext(os.path.basename(file))[0]
            basename = basename.replace("-comp", "")
            path     = 'plots/' + basename + '/'
        else:
            path = 'plots/'

        # Create the folder if it doesn't exist
        if not os.path.isdir(path):
            os.makedirs(path)

        return path

    def save_metadata(self, metadata, save=False):
        """Save metadata info on the path where plots are saved

        Note this method will overwrite the info.txt file. Hence, it should be
        called before all other methods that print to info.txt.

        """

        # Augment the metadata
        duration_ns     = float(self.data[-1]["t1"] - self.data[0]["t1"])
        duration_tdelta = timedelta(microseconds = (duration_ns / 1e3))

        metadata["n_exchanges"] = len(self.data)
        metadata["sync_rate"]   = int(1 / metadata["sync_period"])
        metadata["duration"]    = str(duration_tdelta)

        files = [None]
        if (save):
            files.append(open(self.info, 'w'))

        for f in files:
            print("Setup:", file=f)
            if (f is None):
                print(json.dumps(metadata, indent=4, sort_keys=True))
            else:
                json.dump(metadata, f, indent=4, sort_keys=True)
                print("\n", file=f)

    def _calc_best_case_queueing(self, hops, t_idle, t_fh, t_ptp, direction,
                                 n_rru):
        """Calculate the best-case queueing delay experienced by a PTP frame

        Args:
            hops      : Number of hops the PTP frame traverses
            t_idle    : Idle time between consecutive FH frames
            t_fh      : FH transmission (serialization) delay
            t_ptp     : PTP transmission (serialization) delay
            direction : Message direction (dl or ul)
            n_rru     : Number of RRUs consuming (in DL) or generating (in UL)
                        the FH traffic

        """
        ptp_fh_interval = t_idle # starting interval btw PTP and FH frames
        # Approximation between FH and PTP frames on every hop due to the
        # store-and-forward procedure (which delays the large FH frames more
        # than the small PTP frames)
        approx_per_hop  = (t_fh - t_ptp)

        # Compute the best-case queueing delay iteratively
        b_q_delay = 0
        for i_hop in range(hops):
            # On each store-and-forward hop, approximate the two frames (reduce
            # the interval between them).
            #
            # The exception is in the last hop of the DL direction, when the BBU
            # serves two RRUs. There is a chance that the PTP message (going to
            # RRU1) departs behind a FH frame addressed to RRU2. In this case,
            # the PTP message does not need to wait the FH frame in the last
            # hop. Hence, the two messages do not approximate in this hop. Also,
            # the PTP message does not experience queueing delay in this case.
            if (direction == "dl" and n_rru == 2 and (i_hop + 1) == hops):
                continue
            else:
                ptp_fh_interval -= approx_per_hop

            # Check if the two frames have "touched" each other
            if (ptp_fh_interval < 0 and ptp_fh_interval >= -approx_per_hop):
                # PTP is "reaching" the FH frame in this hop. There will be some
                # queueing delay, but it won't be the full t_fh interval.
                #
                # Example: suppose t_fh is 2.0 us and t_ptp is 0.2 us. Suppose
                # also that the initial interval between them is of 1.0 us
                # (between the end of the FH packet and the beginning of the PTP
                # packet). On initialization, the FH packet departs on instant 0
                # and finishes serialization on instant 2.0 us. The PTP packet
                # departs 1.0 sec later, on instant 3.0 us. If it weren't for
                # the preceding FH frame, the PTP frame would arrive completely
                # on the first hop (switch) on instant 3.2 us and would depart
                # to the next hop. However, the first hop has its output
                # interface occupied by the preceding FH frame until instant 4.0
                # us. As a result, the PTP packet only departs on instant 4.0 us
                # from the switch to the next hop, i.e., 0.8 us later than what
                # it would in the absence of the preceding FH packet. What
                # happened is that in this first hop the PTP-FH approximation
                # was of (2.0 - 0.2) = 1.8 us. However, the starting interval
                # between the end of the FH frame and the start of the PTP frame
                # was only of 1.0 us. So in the first hop, the approximation is
                # larger than the interval between the packets, which means the
                # PTP packet "reaches" the FH packet. The remaining part of the
                # difference "approximation - interval", i.e., of "1.8 - 1.0" is
                # 0.8 us. That's the queuing delay experienced by the PTP packet
                # in this hop.
                b_q_delay += -ptp_fh_interval
            elif (ptp_fh_interval < -approx_per_hop):
                # PTP frame has already "reached" FH frames in previous hops
                b_q_delay += approx_per_hop
                # NOTE: when the PTP message is adjacent to the FH frame, it
                # always experiences a delay of t_fh on each hop. However,
                # "t_fh" is not entirely queueing delay. The queueing delay
                # component is only "t_fh - t_ptp" (equivalent to the
                # approximation per hop), where as the remaining "t_ptp"
                # component is the transmission delay that the PTP frame would
                # normally experience on each store-and-forward hop.
        return b_q_delay

    def calc_expected_delays(self, metadata, save=False):
        """Calculate the expected range of PTP delays in the given FH setup

        Takes queueing, processing, and transmission delays into account.

        Args:
            metadata : Metadata dictionary containing FH setup information
            save     : whether to write results into info.txt

        Returns:
            Dictionary containing (best DL, worst DL, best UL, worst UL)
            theoretical delays including processing and queueing delays.

        """
        # PTP transmission (serialization) delay over 1GbE interface
        t_ptp = 80*8 / 1e9 # PDelayReq/Resp messages are 54 bytes long + plus 26
                           # bytes of Ethernet header (8B preamble + 14B MAC
                           # untagged header + 4B FCS)

        # Overhead bits on FH frames (8B preamble + 18 bytes Ethernet MAC header
        # w/ 802.1Q tag + 12 bytes FH metadata + 2 bytes of stuffing + 4B FCS)
        fh_overhead_bits = (8*44)

        if (metadata["fh_traffic"] is not None):
            l_iq_info  = metadata["fh_traffic"]["iq_size"]
            n_spf_info = metadata["fh_traffic"]["n_spf"]

            if (isinstance(l_iq_info, dict)):
                l_iq_dl = l_iq_info['dl']
                l_iq_ul = l_iq_info['ul']
            else:
                l_iq_dl = l_iq_info
                l_iq_ul = l_iq_info

            if (isinstance(n_spf_info, dict)):
                n_spf_dl = n_spf_info['dl']
                n_spf_ul = n_spf_info['ul']
            else:
                n_spf_dl = n_spf_info
                n_spf_ul = n_spf_info

            frame_size_bits_dl = (l_iq_dl * n_spf_dl) + fh_overhead_bits
            frame_size_bits_ul = (l_iq_ul * n_spf_ul) + fh_overhead_bits

            # Frame transmission (serialization) delay, assuming 1GbE
            t_fh_dl  = frame_size_bits_dl / 1e9
            t_fh_ul  = frame_size_bits_ul / 1e9

            # Fundamental assumption of the ensuing analysis
            assert(t_fh_dl > t_ptp)
            assert(t_fh_ul > t_ptp)

            # Nominal FH frame inter-departure interval
            fs              = metadata["fh_traffic"]["fs"] # sample rate in Hz
            Ts              = 1/fs # sample period in sec
            n_axc_per_frame = 2
            i_fh_dl         = (n_spf_dl / n_axc_per_frame) * Ts
            i_fh_ul         = (n_spf_ul / n_axc_per_frame) * Ts

            # Idle iterval between consecutive FH frames
            n_rru_dl = metadata["fh_traffic"]["n_rru_dl"] if "n_rru_dl" in \
                metadata["fh_traffic"] else metadata["fh_traffic"]["n_rru"]["dl"]
            n_rru_ul  = metadata["fh_traffic"]["n_rru_ul"] if "n_rru_ul" in \
                metadata["fh_traffic"] else metadata["fh_traffic"]["n_rru"]["ul"]
            t_idle_dl = i_fh_dl - (t_fh_dl * n_rru_dl)
            t_idle_ul = i_fh_ul - t_fh_ul

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))
            print("\nExpected theoretical delays:\n", file=files[1])

        # Maintain compatibility with old captures
        if (type(metadata["hops"]) is not dict):
            hops_md = dict()
            hops_md['rru1'] = metadata['hops']
            if (metadata['n_rru_ptp'] == 2):
                hops_md['rru2'] = metadata['hops']
        else:
            hops_md = metadata["hops"]

        expected_delays = dict()
        for dev, hops in hops_md.items():
            # Worst-case and best-case processing delays
            #
            # To compute worst-case queueing delays, we consider the worst-case
            # processing delay per hop. Likewise, for the best-case queueing
            # delays, we consider the best-case processing delay per hop. The
            # values below are based on analysis of some experiments.
            w_p_delay = 3.6e-6 * hops # worst-case processing delay
            b_p_delay = 3.5e-6 * hops # best-case processing delay

            # Transmission delays
            #
            # Corresponds to the store-and-forward processing of the PTP frame
            # on every hop.
            t_delay = t_ptp * hops

            # Worst-case queuing delays in DL and UL
            #
            # The worst-case is when the PTP frame departs already "adjacent"
            # (with no interval) to the preceding FH frame. Consequently, the
            # PTP frame has to wait the entire FH transmission delay on every
            # hop, due to store-and-forward. For example, say the FH
            # transmission delay (t_fh) is 1.0 us, and that the PTP transmission
            # delay (t_ptp) is 0.4 us. Suppose the FH frame departs on instant 0
            # and that the PTP frame departs immediately after, on instant 1.0
            # us. On instant 1.0, the first switch completes reception of the FH
            # frame and can start sending it forward. The output of the first
            # switch remains occupied by the FH frame until instant 2.0. If it
            # were not for the FH frame, the PTP frame would depart on instant
            # 1.0 from the sender and on instant 1.4 from the first switch to
            # the next. However, due to the preceding FH frame, the PTP frame
            # departs only on instant 2.0. As a result, the PTP frame waits an
            # additional 0.6 us (i.e., t_fh - t_ptp) of delay with respect the
            # instant that it would normally depart from the switch (instant
            # 1.4 us). This is the queueing delay component.
            #
            # Also, in the UL, when there are 2 RRUs, and in the tree topology
            # that we typically consider, the PTP frame may need to wait an
            # extra t_fh (FH transmission delay) in the first hop. We assume
            # that this "first hop" is the aggregation switch, which is
            # connected to the two RRUs and has a single shared output to the
            # next hop in the UL direction. Hence, the frames arriving from the
            # two RRUs compete for this shared output. In the worst-case
            # scenario, the PTP frame waits for the full serialization of the FH
            # frame preceding it (coming from the same RRU) plus the
            # serialization of a FH frame from the other RRU that arrived
            # earlier and already holds the outbound interface.
            if (metadata["fh_traffic"] is not None):
                w_dl_q_delay = (t_fh_dl - t_ptp) * hops
                w_ul_q_delay = (t_fh_ul - t_ptp) * hops
                if (n_rru_ul == 2):
                    w_ul_q_delay += t_fh_ul
            else:
                w_dl_q_delay = w_ul_q_delay = 0

            # Best-case queueing delay
            #
            # The best-case is when the PTP frame departs with the maximum
            # interval possible relative to the preceding FH frame. This
            # interval corresponds to the idle interval between two consecutive
            # FH frames. In this scenario, it is possible that the PTP frame
            # does not experience queueing delays for some hops. However, due to
            # the "approximation" between frames caused by the store-and-forward
            # procedure, eventually (after some hops) the PTP frame can still
            # reach the FH frame. From that point on, the PTP frame will
            # experience queueing delays.
            if (metadata["fh_traffic"] is not None):
                b_dl_q_delay = self._calc_best_case_queueing(hops, t_idle_dl,
                                                             t_fh_dl, t_ptp,
                                                             "dl", n_rru_dl)
                b_ul_q_delay = self._calc_best_case_queueing(hops, t_idle_ul,
                                                             t_fh_ul, t_ptp,
                                                             "ul", n_rru_ul)
            else:
                b_dl_q_delay = b_ul_q_delay = 0

            # Total theoretical delays (queueing + transmission + processing)
            w_dl_delay = w_dl_q_delay + t_delay + w_p_delay
            w_ul_delay = w_ul_q_delay + t_delay + w_p_delay
            b_dl_delay = b_dl_q_delay + t_delay + b_p_delay
            b_ul_delay = b_ul_q_delay + t_delay + b_p_delay

            for f in files:
                print("DL FH delay for {}: "
                      "best-case: {:.4f} us "
                      "worst-case: {:.4f} us".format(
                          dev, (b_dl_delay * 1e6), (w_dl_delay * 1e6)
                ), file=f)
                print("UL FH delay for {}: "
                      "best-case: {:.4f} us "
                      "worst-case: {:.4f} us ".format(
                          dev, (b_ul_delay * 1e6), (w_ul_delay * 1e6)
                ), file=f)

            # Save expected delays
            expected_delays[dev] = {
                'b_dl_delay': b_dl_delay,
                'w_dl_delay': w_dl_delay,
                'b_ul_delay': b_ul_delay,
                'w_ul_delay': w_ul_delay
            }

        return expected_delays


    def ptp_exchanges_per_sec(self, save=False):
        """Compute average number of PTP exchanges per second

        Args:
            save    : whether to write results into info.txt

        Returns:
            The computed average

        """
        logger.info("Analyze PTP exchanges per second")
        start_time = self.data[0]["t1"]
        end_time   = self.data[-1]["t1"]
        duration   = float(end_time - start_time)
        n_per_sec  = 1e9 * len(self.data) / duration

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))

        for f in files:
            print("Average no. of PTP exchanges per second: %f" %(n_per_sec),
                  file=f)

        return n_per_sec

    def delay_asymmetry(self, verbose=True, save=False):
        """Analyze the delay asymmetry

        Compute and print some relevant asymmetry metrics.

        Args:
            verbose : whether to print results (otherwise just return it)
            save    : whether to write results into info.txt

        Returns:
            Average delay asymmetry

        """
        logger.info("Analyze delay asymmetry")
        d_asym = np.array([r['asym'] for r in self.data])
        d_ms   = np.array([r["d"] for r in self.data])
        d_sm   = np.array([r["d_bw"] for r in self.data])

        # Mode
        d_ms_mode = stats.mode(np.round(d_ms))[0]
        d_sm_mode = stats.mode(np.round(d_sm))[0]

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))

        if (verbose):
            for f in files:
                print("\nDelay asymmetry analysis:\n", file=f)
                print("Metric \t%12s\t%12s\t%12s" %("m-to-s", "s-to-m",
                                                    "asymmetry"))
                print("Average\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                    np.mean(d_ms), np.mean(d_sm), np.mean(d_asym)), file=f)
                print("Std Dev\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                    np.std(d_ms), np.std(d_sm), np.std(d_asym)), file=f)
                print("Minimum\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                    np.amin(d_ms), np.amin(d_sm),
                    (np.amin(d_ms) - np.amin(d_sm))/2), file=f)
                print("Maximum\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                    np.amax(d_ms), np.amax(d_sm),
                    (np.amax(d_ms) - np.amax(d_sm))/2), file=f)
                print("Median\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                    np.median(d_ms), np.median(d_sm),
                    (np.median(d_ms) - np.median(d_sm))/2), file=f)
                print("Mode\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                    d_ms_mode, d_sm_mode, (d_ms_mode - d_sm_mode)/2), file=f)

        return np.mean(d_asym)

    def _print_err_stats(self, f, key, e, unit):

        mean = np.mean(e)
        sdev = np.std(e)
        rms  = np.sqrt(np.square(e).mean())

        print("{:20s} ".format(key),
              "Mean: {: 8.3f} {} ".format(mean, unit),
              "Sdev: {: 8.3f} {} ".format(sdev, unit),
              "RMS:  {: 8.3f} {}".format(rms, unit), file=f)

    def toffset_err_stats(self, save=False):
        """Print the time offset estimation error statistics

        Args:
            save    : whether to write results into info.txt

        """
        logger.info("Eval time offset estimation error statistics")

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))

        # Skip the transitory (e.g. due to Kalman)
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        for f in files:
            print("\nTime offset estimation error statistics:\n", file=f)

        for suffix, value in est_keys.items():
            key   = "x_est" if (suffix == "raw") else "x_" + suffix
            x_err = [r[key] - r["x"] for r in post_tran_data if key in r]

            if (len(x_err) > 0):
                for f in files:
                    self._print_err_stats(f, key, x_err, "ns")

    def foffset_err_stats(self, save=False):
        """Print the frequency offset estimation error statistics

        Args:
            save    : whether to write results into info.txt

        """
        logger.info("Eval frequency offset estimation error statistics")

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))

        for f in files:
            print("\nFrequency offset estimation error statistics:\n", file=f)

        for suffix, value in est_keys.items():
            key   = "y_est" if (suffix == "raw") else "y_" + suffix
            y_err = [1e9*(r[key] - r["rtc_y"]) for r in self.data
                     if (key in r) and ("rtc_y" in r)]

            if (len(y_err) > 0):
                for f in files:
                    self._print_err_stats(f, key, y_err, "ppb")

    def toffset_drift_err_stats(self, save=False):
        """Print the time offset drift estimation error statistics

        Args:
            save    : whether to write results into info.txt

        """
        logger.info("Eval time offset drift estimation error statistics")

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))

        for f in files:
            print("\nTime offset drift estimation error statistics:\n", file=f)


        # Absolute drift
        true_drift = np.array([(r["x"] - self.data[i-1]["x"])
                               for i,r in enumerate(self.data)
                               if "drift" in r])
        drift_est  = np.array([r["drift"] for r in self.data
                               if "drift" in r])
        drift_err  = drift_est - true_drift

        # Cumulative drift
        true_cum_drift = true_drift.cumsum()
        cum_drif_est   = drift_est.cumsum()
        cum_drif_err   = cum_drif_est - true_cum_drift

        if (len(drift_err) > 0):
            for f in files:
                self._print_err_stats(f, "Drift", drift_err, "ns")
                self._print_err_stats(f, "Cumulative Drift", cum_drif_err, "ns")

    def ranking(self, metric, max_te_win_len=1000, save=False):
        """Rank algorithms based on a chosen performance metric

        Args:
            metric         : Metric used for ranking: max-te, mtie, rms or std
            max_te_win_len : Window length used for the max|TE| computation

        """
        logger.info(f"Rank algorithm performances based on {metric}")

        # Find index where drift estimates start (if any)
        i_drift_start = 0
        for i,r in enumerate(self.data):
            if ("drift" in r):
                i_drift_start = i
                break

        # When drift correction is used, packet selection estimates start only
        # after drift estimate start. Hence, the post-transient dataset must
        # start after drift estimates start. Also, for Kalman filtering results,
        # as a rule of thumb we skip about 20% of the dataset. To accomodate the
        # two requirements, skip the largest of the two:
        n_skip = max(i_drift_start, int(0.2*len(self.data)))
        post_tran_data = self.data[n_skip:]

        # Find the largest number of samples that is guaranteed to be available
        # for all post-processing algorithms that are present in the
        # dataset. For fairness, we will compare all algorithms based on the
        # same number of samples.
        n_samples = np.inf
        for suffix, value in est_keys.items():
            key   = "x_est" if (suffix == "raw") else "x_" + suffix
            x_err = [r[key] - r["x"] for r in post_tran_data if key in r]
            if (len(x_err) > 0):
                n_samples = min(len(x_err), n_samples)
        assert(not np.isinf(n_samples))

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))

        for f in files:
            print(f"\nPerformance ranking based on {metric}:\n", file=f)

        ranking = dict()
        for suffix, value in est_keys.items():
            key   = "x_est" if (suffix == "raw") else "x_" + suffix
            x_err = np.array([r[key] - r["x"] for r in post_tran_data \
                              if key in r])
            x_err = x_err[:n_samples]

            if (len(x_err) ==  0):
                continue

            if (metric == "max-te"):
                if (key in self.results["max_te"]):
                    max_te_est = self.results["max_te"][key][-n_samples:]
                else:
                    max_te_est = self.max_te(x_err, max_te_win_len)
                    self.results["max_te"][key] = max_te_est

                res = max_te_est.mean()

            elif (metric == "mtie"):
                if (key in self.results["mtie"]):
                    _, mtie_est = self.results["mtie"][key]
                    mtie_est    = mtie_est[-n_samples:]
                else:
                    tau_est, mtie_est = self.mtie(x_err)
                    self.results["mtie"][key] = (tau_est, mtie_est)

                res = mtie_est.mean()

            elif (metric == "rms"):
                res = np.sqrt(np.square(x_err).mean())

            elif (metric == "std"):
                res = np.std(x_err)

            else:
                raise ValueError("Metric choice %s unknown" %(metric))

            # Save the result
            ranking[key] = res

        # Print ranking in increasing order
        for key, value in sorted(ranking.items(), key=lambda x: x[1]):
            for f in files:
                print("{:20s}".format(key),
                      "Mean: {: 8.3f} ns".format(value), file=f)

    def check_seq_id_gaps(self, verbose=True, save=False):
        """Check whether there are gaps on sequenceIds"""

        # Print to stdout and, if so desired, to info.txt
        files = [None]
        if (save):
            files.append(open(self.info, 'a'))

        seq_ids = np.array([r["seq_id"] for r in self.data if "seq_id" in r])
        diff    = np.mod(np.diff(seq_ids), 2**16)
        gaps    = np.where(diff != 1)[0]

        # Fix rollovers: apparently our HW (maybe by standard behavior) rolls
        # back to 1, instead of 0.
        non_rollover_gaps = list()
        for i, gap in enumerate(gaps):
            if (not (seq_ids[gap] == 65535 and seq_ids[gap+1] == 1)):
                non_rollover_gaps.append(gap)
            elif (verbose):
                logging.debug("Gap from {:d} to {:d} due to rollover".format(
                    seq_ids[gap], seq_ids[gap+1]))

        if (len(seq_ids) == 0):
            logger.warning("Dataset doesn't contain sequenceIds")

        for f in files:
            if (len(non_rollover_gaps) > 0 and verbose):
                print("sequenceId gaps identified: {:d}".format(
                    len(non_rollover_gaps)), file=f)
                for gap in non_rollover_gaps:
                    logging.debug("Gap from {:d} to {:d}".format(
                        seq_ids[gap], seq_ids[gap+1]))
            elif (verbose):
                print("Checking sequenceIDs: OK (no gaps)", file=f)

        if (len(non_rollover_gaps) > 0):
            raise ValueError("Dataset has sequenceId gaps")

    def rolling_window_mtx(self, x, window_size, shift=1):
        """Compute all overlapping (rolling) observation windows in a matrix

        Args:
            x           : observation vector that is supposed to be split into
                          overlapping windows
            shift       : Controls the shift between consecutive windows or,
                          equivalently, the overlap. For instance, if shift=1,
                          each window overlaps with N-1 samples of the previous
                          window. If shift=window_size, the windows are
                          completely disjoint.
            window_size : the target window size

        Returns:

            Window matrix with all windows as rows. That is, if n_windows is the
            number of windows, the result has dimensions:

            (n_windows, window_size)

        """
        if window_size < 1:
            raise ValueError("`window_size` must be at least 1.")
        if window_size > x.shape[-1]:
            raise ValueError("`window_size` is too long.")

        shape   = x.shape[:-1] + (x.shape[-1] - window_size + 1, window_size)
        strides = x.strides + (x.strides[-1],)

        return np.lib.stride_tricks.as_strided(x, shape=shape,
                                               strides=strides)[0::shift]

    def mtie(self, te, window_step = 2, starting_window = 16):
        """Maximum time interval error (MTIE)

        Computes the MTIE based on time error (TE) samples. The MTIE computes
        the peak-to-peak time interval error (TIE) over windows of increasing
        duration.

        Args:
            te              : Vector of TE values
            window_step     : Enlarge window by this step on every iteration
            starting_window : Starting window size

        Returns:
            tau_array  : MTIE observation intervals
            mtie_array : The calculated MTIE for each observation interval

        """
        assert(isinstance(te, np.ndarray))

        n_samples = len(te) # total number of samples

        # Number of different intervals to be evaluated
        log_max_win_size   = math.floor(math.log2(n_samples/2))
        max_win_size       = 2**log_max_win_size
        log_start_win_size = math.floor(math.log2(starting_window))
        n_tau              = log_max_win_size - log_start_win_size + 1

        # Preallocate results
        mtie_array    = np.zeros(n_tau)
        tau_array     = np.zeros(n_tau)

        # Try until the window occupies half of the data length, so that the
        # maximum window size still fits twice on the data
        i_tau       = 0
        window_size = starting_window
        while (window_size <= max_win_size):
            # Get all possible windows with the current window size:
            parted_array = self.rolling_window_mtx(te, window_size)

            # Get maximum and minimum TE values of each window
            window_max = np.max(parted_array, axis = 1)
            window_min = np.min(parted_array, axis = 1)

            # MTIE candidates (maximum TIE of each window):
            mtie_candidates = window_max - window_min

            # Final MTIE is the maximum among all candidates
            mtie = np.amax(mtie_candidates)

            # Save MTIE and current window duration within outputs
            mtie_array[i_tau] = mtie
            tau_array[i_tau]  = window_size

            # Update window size
            window_size = window_size * window_step

            i_tau += 1

        # Have all expected tau values been evaluated?
        assert(n_tau == i_tau), \
            "n_tau = %d, i_tau = %d" %(n_tau, i_tau)

        return tau_array, mtie_array

    def max_te(self, te, window_len):
        """Maximum absolute time error (max|TE|)

        Computes the max|TE| based on time error (TE) samples. The max|TE|
        metric compute the maximum among the absolute time error sample over
        a sliding non-overlapping window.

        Args:
            window_len : Window length
            te         : Vector of time error (TE) values

        Returns:
            max_te     : The calculated Max|TE| over a sliding window

        """
        assert(isinstance(te, np.ndarray))

        te_mtx = self.rolling_window_mtx(te, window_len, shift=window_len)
        max_te = np.amax(np.abs(te_mtx), axis=1)

        return max_te

    def dec_plot_filter(func):
        """Plot filter decorator

        Filter the global 'est_keys' dict based on 'show_' args from
        'plot_' functions.

        Args:
            func = Plot functions

        """
        def wrapper(*args, **kwargs):
            for k, v in kwargs.items():
                if (not v):
                    # Extract the preffix_keys from 'show_' variables
                    prefix_re  = (re.search(r'(?<=show_).*', k))
                    if (prefix_re is None):
                        continue
                    prefix_key = prefix_re.group(0)
                    # Find the dict keys that match with the preffix_keys
                    key_values  = [key for key in est_keys if
                                   re.match(r'^{}_*'.format(prefix_key), key)]
                    # Set show key to 'False' on global 'est_keys' dict
                    for suffix, v in est_keys.items():
                        if (suffix in key_values):
                            v["show"] = False

            # Run plot function
            plot_function = func(*args, **kwargs)

            # Clean-up global 'est_keys' dict after running the plot function
            for k, v in est_keys.items():
                v["show"] = True

            return plot_function
        return wrapper

    @dec_plot_filter
    def plot_toffset_vs_time(self, show_raw=True, show_best=True, show_ls=True,
                             show_pkts=True, show_kf=True, show_loop=True,
                             show_true=True, n_skip=None, x_unit='time',
                             save=True, save_format=None, dpi=None):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw    : Show raw measurements
            show_best   : Enable to highlight the best measurements.
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            show_loop   : Show PI loop fit
            n_skip      : Number of initial samples to skip
            show_true   : Show true values
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot time offset vs. time")
        n_skip         = int(0.2*len(self.data)) if (not n_skip) else n_skip
        post_tran_data = self.data[n_skip:]

        # Time axis
        t_start  = post_tran_data[0]["t1"]
        time_vec = np.array([float(r["t1"] - t_start) for r in post_tran_data])\
                   / NS_PER_MIN

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            x_axis_label = 'Time (min)'
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        plt.figure(figsize=self.figsize)

        for suffix, value in est_keys.items():
            if (value["show"]):
                if (suffix == "raw"):
                    key = "x_est"
                elif (suffix == "true"):
                    key = "x"
                else:
                    key = "x_" + suffix

                x_est = [r[key] for r in post_tran_data if key in r]

                # Define the x axis - either in time or in samples
                if (x_unit == "time"):
                    x_axis_vec   = [time_vec[i] for i, r in
                                    enumerate(post_tran_data) if key in r]
                elif (x_unit == "samples"):
                    x_axis_vec   = [r["idx"] for r in post_tran_data
                                    if key in r]

                if (len(x_est) > 0):
                    plt.scatter(x_axis_vec, x_est,
                                label=value["label"], marker=value["marker"],
                                s=1.0)

        # Best raw measurements
        if (show_best):
            x_tilde  = np.array([r["x_est"] for r in post_tran_data])
            x        = np.array([r["x"] for r in post_tran_data])

            # Find best raw measurements (with error under 10 ns)
            err      = x_tilde - x
            best_idx = np.squeeze(np.where(abs(err) < 10))

            # Define the x axis - either in time or in samples
            if (x_unit == "time"):
                x_axis_vec   = time_vec[best_idx]
            elif (x_unit == "samples"):
                x_axis_vec   = best_idx + n_skip

            plt.scatter(x_axis_vec, x_tilde[best_idx], s=1.0,
                        label="Accurate Measurements")

        plt.xlabel(x_axis_label)
        plt.ylabel('Time offset (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "toffset_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_toffset_err_vs_time(self, show_raw=True, show_ls=True,
                                 show_pkts=True, show_kf=True, show_loop=True,
                                 n_skip=None, x_unit='time', save=True,
                                 save_format=None, dpi=None):
        """Plot time offset error vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show packet selection fit
            show_kf     : Show Kalman filtering results
            show_loop   : Show PI loop fit
            n_skip      : Number of initial samples to skip
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot time offset estimation error vs. time")
        n_skip         = int(0.2*len(self.data)) if (not n_skip) else n_skip
        post_tran_data = self.data[n_skip:]

        # Time axis
        t_start  = post_tran_data[0]["t1"]
        time_vec = np.array([float(r["t1"] - t_start) for r in post_tran_data])\
                   / NS_PER_MIN

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            x_axis_label = 'Time (min)'
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        plt.figure(figsize=self.figsize)

        for suffix, value in est_keys.items():
            if (value["show"]):
                key   = "x_est" if (suffix == "raw") else "x_" + suffix
                x_err = [r[key] - r["x"] for r in post_tran_data if key in r]

                # Define the x axis - either in time or in samples
                if (x_unit == "time"):
                    x_axis_vec = [time_vec[i] for i, r in
                                  enumerate(post_tran_data) if key in r]
                elif (x_unit == "samples"):
                    x_axis_vec = [r["idx"] for r in post_tran_data if key in r]

                if (len(x_err) > 0):
                    plt.scatter(x_axis_vec, x_err, label=value["label"],
                                marker=value["marker"], s=20.0, alpha=0.7)

        plt.xlabel(x_axis_label)
        plt.ylabel('Time offset Error (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "toffset_err_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_toffset_err_hist(self, show_raw=True, show_ls=True, show_pkts=True,
                              show_kf=True, show_loop=True, n_bins=50,
                              save=True, save_format=None, dpi=None):
        """Plot time offset error histogram

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show packet selection fit
            show_kf     : Show Kalman filtering results
            show_loop   : Show PI loop fit
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot time offset estimation error histogram")
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        plt.figure(figsize=self.figsize)

        for suffix, value in est_keys.items():
            if (value["show"]):
                key   = "x_est" if (suffix == "raw") else "x_" + suffix
                x_err = [r[key] - r["x"] for r in post_tran_data if key in r]

                if (len(x_err) > 0):
                    plt.hist(x_err, bins=50, density=True, alpha=0.7,
                             label=value["label"])

        plt.xlabel('Time offset Error (ns)')
        plt.ylabel('Probability Density')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "toffset_err_hist."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_delay_hist(self, show_raw=True, show_true=True, n_bins=50,
                        split=False, save=True, save_format=None, dpi=None):
        """Plot delay histogram

        Plot histogram of delays in microseconds.

        Args:
            show_raw    : Show histogram of raw delay measurements
            show_true   : Show histogram true delay values
            n_bins      : Target number of bins
            split       : Whether to split each delay histogram (m-to-s, s-to-m
                          and estimate) on a different figure
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot delay histogram")
        x_label = 'Delay (us)'
        y_label = 'Probability Density'

        if (split):
            plots   = list()

            if (show_raw):
                d_est = np.array([r['d_est'] for r in self.data]) / 1e3
                plt.figure(figsize=self.figsize)
                plt.hist(d_est, bins=n_bins, density=True)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.grid(color='k', linewidth=.5, linestyle=':')
                plt.title("Two-way Measurements")
                plots.append({"plt" : plt.gcf(),
                              "label" : "raw"})

            if (show_true):
                d = np.array([r["d"] for r in self.data]) / 1e3
                d_bw = np.array([r["d_bw"] for r in self.data]) / 1e3

                plt.figure(figsize=self.figsize)
                plt.hist(d, bins=n_bins, density=True)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.grid(color='k', linewidth=.5, linestyle=':')
                plt.title("True master-to-slave")
                plots.append({"plt": plt.gcf(),
                              "label": "m2s"})

                plt.figure(figsize=self.figsize)
                plt.hist(d_bw, bins=n_bins, density=True)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.grid(color='k', linewidth=.5, linestyle=':')
                plt.title("True slave-to-master")
                plots.append({"plt": plt.gcf(),
                              "label": "s2m"})

            for p in plots:
                if (save):
                    img_name   = self.prefix + "delay_hist_" + p["label"] + "."
                    img_format = save_format or self.save_format
                    img_dpi    = dpi or self.dpi
                    p["plt"].tight_layout()
                    p["plt"].savefig(self.path + img_name + img_format,
                                     format=img_format, dpi=img_dpi)
                else:
                    p["plt"].show()
        else:
            # Single plot
            plt.figure(figsize=self.figsize)
            if (show_raw):
                d_est = np.array([r['d_est'] for r in self.data]) / 1e3
                plt.hist(d_est, bins=n_bins, density=True, alpha=0.5,
                         label="Two-way Measurements")

            if (show_true):
                d = np.array([r["d"] for r in self.data]) / 1e3
                plt.hist(d, bins=n_bins, density=True, alpha=0.5,
                         label="True master-to-slave")
                d_bw = np.array([r["d_bw"] for r in self.data]) / 1e3
                plt.hist(d_bw, bins=n_bins, density=True, alpha=0.5,
                         label="True slave-to-master")

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(color='k', linewidth=.5, linestyle=':')
            plt.legend()

            if (save):
                img_name   = self.prefix + "delay_hist."
                img_format = save_format or self.save_format
                img_dpi    = dpi or self.dpi
                plt.tight_layout()
                plt.savefig(self.path + img_name + img_format,
                            format=img_format, dpi=img_dpi)
            else:
                plt.show()
            plt.close()

    def plot_delay_vs_time(self, x_unit='time', show_raw=True, split=False,
                           save=True, save_format=None, dpi=None):
        """Plot delay estimations vs time

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            show_raw    : Show raw measurements
            split       : Whether to split m-to-s and s-to-m plots
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot delay vs. time")
        n_data = len(self.data)

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            t_start      = self.data[0]["t1"]
            x_axis_vec   = np.array([float(r["t1"] - t_start) for r in \
                                     self.data]) / NS_PER_MIN
            x_axis_label = 'Time (min)'

        elif (x_unit == "samples"):
            x_axis_vec   = range(0, n_data)
            x_axis_label = 'Realization'

        d      = [r["d"] for r in self.data]
        d_bw   = [r["d_bw"] for r in self.data]

        if (show_raw):
            d_est  = [r["d_est"] for r in self.data]

        if (split):
            plt.figure(figsize=self.figsize)
            plt.scatter(x_axis_vec, d, s = 1.0)
            plt.xlabel(x_axis_label)
            plt.ylabel('m-to-s delay (ns)')
            plt.grid(color='k', linewidth=.5, linestyle=':')

            if (save):
                img_name   = self.prefix + "delay_m2s_vs_time."
                img_format = save_format or self.save_format
                img_dpi    = dpi or self.dpi
                plt.tight_layout()
                plt.savefig(self.path + img_name + img_format,
                            format=img_format, dpi=img_dpi)
            else:
                plt.show()
            plt.close()

            plt.figure(figsize=self.figsize)
            plt.scatter(x_axis_vec, d_bw, s = 1.0)
            plt.xlabel(x_axis_label)
            plt.ylabel('s-to-m delay (ns)')
            plt.grid(color='k', linewidth=.5, linestyle=':')

            if (save):
                img_name   = self.prefix + "delay_s2m_vs_time."
                img_format = save_format or self.save_format
                img_dpi    = dpi or self.dpi
                plt.tight_layout()
                plt.savefig(self.path + img_name + img_format,
                            format=img_format, dpi=img_dpi)
            else:
                plt.show()
            plt.close()
        else:
            plt.figure(figsize=self.figsize)
            if (show_raw):
                plt.scatter(x_axis_vec, d_est, label="Raw Measurements", s = 1.0)
            plt.scatter(x_axis_vec, d, label="True Values", s = 1.0)
            plt.xlabel(x_axis_label)
            plt.ylabel('Delay Estimation (ns)')
            plt.grid(color='k', linewidth=.5, linestyle=':')
            plt.legend()

            if (save):
                img_name   = self.prefix + "delay_vs_time."
                img_format = save_format or self.save_format
                img_dpi    = dpi or self.dpi
                plt.tight_layout()
                plt.savefig(self.path + img_name + img_format,
                            format=img_format, dpi=img_dpi)
            else:
                plt.show()
            plt.close()

    def plot_delay_est_err_vs_time(self, x_unit='time', save=True,
                                   save_format=None, dpi=None):
        """Plot delay estimations error vs time

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot delay estimation error vs. time")
        n_data    = len(self.data)

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            t_start      = self.data[0]["t1"]
            x_axis_vec   = np.array([float(r["t1"] - t_start) for r in \
                                     self.data]) / NS_PER_MIN
            x_axis_label = 'Time (min)'

        elif (x_unit == "samples"):
            x_axis_vec   = range(0, n_data)
            x_axis_label = 'Realization'

        d_est_err = [r["d_est"] - r["d"] for r in self.data]

        plt.figure(figsize=self.figsize)
        plt.scatter(x_axis_vec, d_est_err, s = 1.0)
        plt.xlabel(x_axis_label)
        plt.ylabel('Delay Estimation Error (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')

        if (save):
            img_name   = self.prefix + "delay_est_err_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name  + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_delay_asym_hist(self, n_bins=50, save=True, save_format=None,
                             dpi=None):
        """Plot delay asymmetry histogram

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot delay asymmetry histogram")

        plt.figure(figsize=self.figsize)
        d_asym = np.array([r['asym'] for r in self.data]) / 1e3
        plt.hist(d_asym, bins=n_bins, density=True)
        plt.xlabel('Delay asymmetry (us)')
        plt.ylabel('Probability Density')
        plt.grid(color='k', linewidth=.5, linestyle=':')

        if (save):
            img_name   = self.prefix + "delay_asym_hist."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_delay_asym_vs_time(self, save=True, x_unit='time',
                                save_format=None, dpi=None):
        """Plot delay asymmetry over time

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot delay asymmetry vs. time")

        # Time axis
        t_start  = self.data[0]["t1"]
        time_vec = np.array([float(r["t1"] - t_start) for r in self.data])\
                   / NS_PER_MIN

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            x_axis_label = 'Time (min)'
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        d_asym = np.array([r['asym'] for r in self.data]) / 1e3

        plt.figure(figsize=self.figsize)
        plt.scatter(time_vec, d_asym, s=1.0)
        plt.xlabel(x_axis_label)
        plt.ylabel('Delay asymmetry (us)')
        plt.grid(color='k', linewidth=.5, linestyle=':')

        if (save):
            img_name   = self.prefix + "delay_asym_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_foffset_vs_time(self, show_raw=True, show_ls=True, show_kf=True,
                             show_true=True, n_skip=None, x_unit='time',
                             save=True, save_format=None, dpi=None):
        """Plot freq. offset vs time

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares estimations
            show_kf     : Show Kalman filtering results
            n_skip      : Number of initial samples to skip
            show_true   : Show true values
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """

        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot frequency offset vs. time")
        n_skip         = int(0.2*len(self.data)) if (not n_skip) else n_skip
        post_tran_data = self.data[n_skip:]

        # Time axis
        t_start  = post_tran_data[0]["t1"]
        time_vec = np.array([float(r["t1"] - t_start) for r in post_tran_data])\
                   / NS_PER_MIN

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            x_axis_label = 'Time (min)'
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        plt.figure(figsize=self.figsize)

        for suffix, value in est_keys.items():
            if (value["show"]):
                if (suffix == "raw"):
                    key   = "y_est"
                elif (suffix == "true"):
                    key   = "rtc_y"
                else:
                    key   = "y_" + suffix

                # Get the normalized frequency offset values and convert to ppb
                y_est_ppb = [1e9*r[key] for r in post_tran_data if key in r]

                # Define the x axis - either in time or in samples
                if (x_unit == "time"):
                    x_axis_vec   = [time_vec[i] for i, r in
                                    enumerate(post_tran_data) if key in r]
                elif (x_unit == "samples"):
                    x_axis_vec   = [r["idx"] for r in post_tran_data if key in r]

                if (len(y_est_ppb) > 0):
                    plt.scatter(x_axis_vec, y_est_ppb,
                                label=value["label"], marker=value["marker"],
                                s=1.0)

        plt.xlabel(x_axis_label)
        plt.ylabel('Frequency Offset (ppb)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "foffset_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_foffset_err_vs_time(self, show_raw=True, show_ls=True,
                                 show_kf=True, n_skip_kf=0, x_unit='time',
                                 save=True, save_format=None, dpi=None):
        """Plot freq. offset estimation error vs time

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares estimations
            show_kf     : Show Kalman filtering results
            n_skip_kf   : Number of initial Kalman filter samples to skip
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """

        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot frequency offset estimation error vs. time")
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        # Time axis
        t_start  = post_tran_data[0]["t1"]
        time_vec = np.array([float(r["t1"] - t_start) for r in post_tran_data])\
                   / NS_PER_MIN

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            x_axis_label = 'Time (min)'
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        plt.figure(figsize=self.figsize)

        for suffix, value in est_keys.items():
            if (value["show"]):
                if (suffix == "raw"):
                    key   = "y_est"
                else:
                    key   = "y_" + suffix

                # Get the normalized frequency offset values and convert to ppb
                y_est_err_ppb = [1e9*(r[key] - r["rtc_y"]) for r
                                 in post_tran_data if key in r]

                # Define the x axis - either in time or in samples
                if (x_unit == "time"):
                    x_axis_vec   = [time_vec[i] for i, r in
                                    enumerate(post_tran_data) if key in r]
                elif (x_unit == "samples"):
                    x_axis_vec   = [r["idx"] for r in post_tran_data if key in r]

                if (len(y_est_err_ppb) > 0):
                    plt.scatter(x_axis_vec, y_est_err_ppb,
                                label=value["label"], marker=value["marker"],
                                s=1.0)

        plt.xlabel(x_axis_label)
        plt.ylabel('Frequency Offset Error (ppb)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "foffset_err_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_foffset_err_hist(self, show_raw=True, show_ls=True, show_kf=True,
                              n_bins=50, save=True, save_format=None, dpi=None):
        """Plot frequency offset error histogram

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_kf     : Show Kalman filtering results
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot time offset estimation error histogram")
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        plt.figure(figsize=self.figsize)

        for suffix, value in est_keys.items():
            if (value["show"]):
                key   = "y_est" if (suffix == "raw") else "y_" + suffix
                y_err = [1e9*(r[key] - r["rtc_y"]) for r in post_tran_data
                         if key in r]

                if (len(y_err) > 0):
                    plt.hist(y_err, bins=50, density=True, alpha=0.7,
                             label=value["label"])

        plt.xlabel('Frequency Offset Error (ppb)')
        plt.ylabel('Probability Density')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "foffset_err_hist."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_pdv_vs_time(self, x_unit='time', save=True, save_format=None,
                         dpi=None):
        """Plot PDV over time

        Each plotted value represents the measured difference of the current
        Sync/DelayReq delay with respect to the delay experienced by the
        previous Sync/DelayReq. Note that the actual delay is not measurable,
        but the difference in delay between consecutive messages is.

        Timestamp differences can be used for this computation. Recall they are:

        t_{2,1}[n] = t_2[n] - t_1[n] = x[n] + d_{ms}[n]
        t_{4,3}[n] = t_4[n] - t_3[n] = -x[n] + d_{sm}[n]

        Thus, by observing the diff of $t_{2,1}[n]$, for example, we get:

        t_{2,1}[n] - t_{2,1}[n-1] = (x[n] - x[n-1]) + (d_{ms}[n] - d_{ms}[n-1])

        Since $x[n]$ varies slowly, this is approximately equal to:

        t_{2,1}[n] - t_{2,1}[n-1] \approx d_{ms}[n] - d_{ms}[n-1]

        Similarly, in the reverse direction, we have:

        t_{4,3}[n] - t_{4,3}[n-1] \approx d_{sm}[n] - d_{sm}[n-1]

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot PDV vs. time")
        n_data  = len(self.data)

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            t_start      = self.data[0]["t1"]
            x_axis_vec   = np.array([float(r["t1"] - t_start) for r in \
                                     self.data]) / NS_PER_MIN
            x_axis_label = 'Time (min)'

        elif (x_unit == "samples"):
            x_axis_vec   = range(0, n_data)
            x_axis_label = 'Realization'

        # Timestamp differences
        t2_1 = np.array([float(r["t2"] - r["t1"]) for r in self.data])
        t4_3 = np.array([float(r["t4"] - r["t3"]) for r in self.data])

        # Diffs
        diff_t2_1 = np.diff(t2_1)
        diff_t4_3 = np.diff(t4_3)

        plt.figure(figsize=self.figsize)
        plt.scatter(x_axis_vec[1:], diff_t2_1, s = 1.0, label="m-to-s")
        plt.scatter(x_axis_vec[1:], diff_t4_3, s = 1.0, label="s-to-m")
        plt.xlabel(x_axis_label)
        plt.ylabel('Delay Variation (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "pdv_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_pdv_hist(self, n_bins=50, save=True, save_format=None, dpi=None):
        """Plot PDV histogram

        See explanation of "plot_pdv_vs_time".

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot PDV histogram")

        # Timestamp differences
        t2_1 = np.array([float(r["t2"] - r["t1"]) for r in self.data])
        t4_3 = np.array([float(r["t4"] - r["t3"]) for r in self.data])

        # Diffs
        diff_t2_1 = np.diff(t2_1)
        diff_t4_3 = np.diff(t4_3)

        plt.figure(figsize=self.figsize)
        plt.hist(diff_t2_1, bins=n_bins, density=True, alpha=0.7,
                 label="m-to-s")
        plt.hist(diff_t4_3, bins=n_bins, density=True, alpha=0.7,
                 label="s-to-m")
        plt.xlabel('Delay Variation (ns)')
        plt.ylabel('Probability Density')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "pdv_hist."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_ptp_exchange_interval_vs_time(self, n_bins=200, save=True,
                                           save_format=None, dpi=None):
        """Plot CDF of the interval between PTP exchanges
        """

        logger.info("Plot PTP exchange interval vs. time")

        # Exchange interval
        for t in ["t1", "t2", "t3", "t4"]:
            t_diff = np.diff(np.array([float(r[t]) for r in self.data]))/1e6
            plt.figure(figsize=self.figsize)
            plt.hist(t_diff, bins=n_bins, density=True, cumulative=True,
                     histtype='step', alpha=0.8, color='k')
            plt.xlabel('${0}[n] - {0}[n-1]$ (ms)'.format(t))
            plt.ylabel('CDF')
            plt.title('PTP exchange interval')
            plt.grid(color='k', linewidth=.5, linestyle=':')

            if (save):
                img_name   = self.prefix + f"{t}_delta_vs_time."
                img_format = save_format or self.save_format
                img_dpi    = dpi or self.dpi
                plt.tight_layout()
                plt.savefig(self.path + img_name+ img_format,
                            format=img_format, dpi=img_dpi)
            else:
                plt.show()
            plt.close()

    def plot_toffset_drift_vs_time(self, x_unit='time', save=True,
                                   save_format=None, dpi=None):
        """Plot time offset drift vs. time

        It is useful to analyze how x[n] varies between consecutive PTP
        exchanges. This plot shows (x[n] - x[n-1]) over time.

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot time offset drift vs. time")
        n_data  = len(self.data)

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            t_start      = self.data[0]["t1"]
            x_axis_vec   = np.array([float(r["t1"] - t_start) for r in \
                                     self.data if "drift" in r]) / NS_PER_MIN
            x_axis_label = 'Time (min)'

        elif (x_unit == "samples"):
            x_axis_vec   = range(0, n_data)
            x_axis_label = 'Realization'

        true_drift = np.array([(r["x"] - self.data[i-1]["x"])
                               for i,r in enumerate(self.data)
                               if "drift" in r])
        drift_est  = np.array([r["drift"] for r in self.data if "drift" in r])

        plt.figure(figsize=self.figsize)
        plt.scatter(x_axis_vec, true_drift, s = 1.0, label="True")
        plt.scatter(x_axis_vec, drift_est, s = 1.0, label="Estimate")
        plt.xlabel(x_axis_label)
        plt.ylabel('$x[n] - x[n-1]$ (ns)')
        plt.title('Time offset drift')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "toffset_drift_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name  + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

        cum_drift_err = drift_est.cumsum() - true_drift.cumsum()

        plt.figure(figsize=self.figsize)
        plt.scatter(x_axis_vec, cum_drift_err, s = 1.0)
        plt.xlabel(x_axis_label)
        plt.ylabel('(ns)')
        plt.title('Cumulative time offset drift error')

        if (save):
            img_name   = self.prefix + "toffset_drift_cum_err_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_toffset_drift_hist(self, n_bins=50, save=True, save_format=None,
                                dpi=None):
        """Plot time offset drift histogram

        Histogram of the drift (x[n] - x[n-1]).

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot time offset drift histogram")

        # True time offset
        x = np.array([r["x"] for r in self.data])

        plt.figure(figsize=self.figsize)
        plt.hist(np.diff(x), bins=n_bins, density=True)
        plt.xlabel('$x[n] - x[n-1]$ (ns)')
        plt.ylabel('Probability Density')
        plt.title('True time offset drift histogram')
        plt.grid(color='k', linewidth=.5, linestyle=':')

        if (save):
            img_name   = self.prefix + "toffset_drift_hist."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_mtie(self, show_raw=True, show_ls=True, show_pkts=True,
                  show_kf=True, show_loop=True, save=True, save_format=None,
                  dpi=None):
        """Plot MTIE versus the observation interval(Tau)

        Plots MTIE. The time error (TE) samples are assumed to be equal to the
        time offset estimation errors. The underlying assumption is that in
        practice these estimations would be used to correct the clock and thus
        the resulting TE would correspond to the residual error of the imperfect
        (noisy) time offset estimation. The time interval error (TIE), in turn,
        is the difference between TE samples over a given interval. Essentially,
        it assesses how accurately the RTC can measure elapsed intervals. If the
        time offset is constant during an interval measurement, i.e., the ending
        TE is the same as the starting TE, the TIE is zero, meaning the RTC
        measures the interval perfectly. The TE variations experienced by the
        clock during an interval are the reason why the corresponding interval
        measurements are noisy and, thus, there is TIE.

        In our implementation, the observation window durations of the
        associated TIE samples are assumed to be given in terms of number of
        samples that they contain, rather than their actual time durations. This
        is not strictly how MTIE is computed, but useful for the evaluation and
        simpler to implement.

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            show_loop   : Show PI loop fit
            n_skip      : Number of initial samples to skip
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot MTIE")
        plt.figure(figsize=self.figsize)

        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        for suffix, value in est_keys.items():
            if (not value["show"] or suffix == "true"):
                continue

            key = "x_est" if (suffix == "raw") else "x_" + suffix

            if (key in self.results["mtie"]):
                # Take mtie from cached results
                tau_est, mtie_est = self.results["mtie"][key]
            else:
                x_err = np.array([r[key] - r["x"] for r in post_tran_data \
                                  if key in r])

                if (len(x_err) == 0):
                    continue

                tau_est, mtie_est = self.mtie(x_err)

                # Add to cached results
                self.results["mtie"][key] = (tau_est, mtie_est)

            plt.semilogx(tau_est, mtie_est, label=value["label"],
                         marker=value["marker"], alpha=0.7, basex=2)

        plt.xlabel('Observation interval (samples)')
        plt.ylabel('MTIE (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend(loc=0)

        if (save):
            img_name   = self.prefix + "mtie_vs_tau."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_max_te(self, window_len, show_raw=True, show_ls=True,
                    show_pkts=True, show_kf=True, show_loop=True, n_skip=None,
                    x_unit='time', save=True, save_format=None, dpi=None):
        """Plot Max|TE| vs time.

        Args:
            window_len  : Window lengths
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            show_loop   : Show PI loop fit
            n_skip      : Number of initial samples to skip
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot max|TE| vs. time")
        n_skip         = int(0.2*len(self.data)) if (not n_skip) else n_skip
        post_tran_data = self.data[n_skip:]

        # Time axis
        t_start  = post_tran_data[0]["t1"]
        time_vec = np.array([float(r["t1"] - t_start) for r in post_tran_data])\
                   / NS_PER_MIN

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            x_axis_label = 'Time (min)'
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        # Dynamic markers
        marker = itertools.cycle(g_markers)

        plt.figure(figsize=self.figsize)

        for suffix, value in est_keys.items():
            if (not value["show"] or suffix == "true"):
                continue

            key = "x_est" if (suffix == "raw") else "x_" + suffix

            if (key in self.results["max_te"]):
                # Take max|TE| from cached results
                max_te_est = self.results["max_te"][key]
            else:
                te = np.array([r[key] - r["x"] for r in post_tran_data \
                               if key in r])

                if (len(te) == 0 or len(te) < window_len):
                    continue

                max_te_est = self.max_te(te, window_len)

                # Add to cached results
                self.results["max_te"][key] = max_te_est

            # Define the x axis - either in time or in samples
            if (x_unit == "time"):
                x_axis_vec = [time_vec[i] for i, r in
                              enumerate(post_tran_data) if key in r]
            elif (x_unit == "samples"):
                x_axis_vec = [r["idx"] for r in post_tran_data if key in r]

            # If the latex interpreter is being used, try to break the
            # legends to fit in two lines. This approach will help to save
            # spacing.
            if (self.usetex):
                label = value["label"].replace(' ', '\n').replace('-', '\n')
            else:
                label = value["label"]

            plt.plot(x_axis_vec[window_len - 1::window_len], max_te_est,
                     label=label, markersize=3, marker = next(marker))

        plt.xlabel(x_axis_label)
        plt.ylabel('$\max|$TE$|$ (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')

        if (self.usetex):
            plt.legend(fontsize='small', bbox_to_anchor=(1.0, 1.02),
                       loc='upper left', frameon=False, prop={'size': 5.})
        else:
            plt.legend(loc=0)

        if (save):
            img_name   = self.prefix + "max_te_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_temperature(self, x_unit='time', save=True, save_format=None,
                         dpi=None):
        """Plot temperature vs time

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot temperature")

        # TODO: move the definition of x-axis label into the decorator
        if (x_unit == "time"):
            t_start      = self.data[0]["t1"]
            time_vec     = np.array([float(r["t1"] - t_start) for r in \
                                    self.data]) / NS_PER_MIN
            x_axis_vec   = [time_vec[i] for i, r in enumerate(self.data) \
                            if "temp" in r]
            x_axis_label = 'Time (min)'

        elif (x_unit == "samples"):
            x_axis_vec   = [r["idx"] for r in self.data if "temp" in r]
            x_axis_label = 'Realization'

        temp1 = [r["temp"][0] for r in self.data if "temp" in r]
        temp2 = [r["temp"][1] for r in self.data if "temp" in r]

        plt.figure(figsize=self.figsize)
        plt.scatter(x_axis_vec, temp1, s = 1.0, label="LM35")
        plt.scatter(x_axis_vec, temp2, s = 1.0, label="MCP9808")
        plt.xlabel(x_axis_label)
        plt.ylabel('Temperature (C)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "temperature_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_occupancy(self, x_unit='time', save=True, save_format=None,
                       dpi=None):
        """Plot BBU/RRU DAC interface buffer occupancy vs time

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'
            dpi         : Image resolution in dots per inch

        """
        logger.info("Plot occupancy")

        if (x_unit == "time"):
            t_start      = self.data[0]["t1"]
            x_axis_label = 'Time (min)'
            time_vec     = np.array([float(r["t1"] - t_start) for r in \
                                     self.data]) / NS_PER_MIN
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        # Plot BBU and RRU occupancies
        plt.figure(figsize=self.figsize)
        for key, label in [("rru_occ", "RRU"), ("bbu_occ", "BBU"),
                           ("rru2_occ", "RRU2")]:
            if (x_unit == "time"):
                x_axis_vec   = [time_vec[i] for i, r in enumerate(self.data) \
                                if key in r and isinstance(r[key], int)]
            elif (x_unit == "samples"):
                x_axis_vec   = [r["idx"] for r in self.data if key in r and
                                isinstance(r[key], int)]

            occ = np.array([int(r[key]) for r in self.data if key in r and
                            isinstance(r[key], int)])
            plt.scatter(x_axis_vec, occ, s = 1.0, label=label)

        plt.xlabel(x_axis_label)
        plt.ylim((0, 8191))
        plt.ylabel('Occupancy')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + "occupancy_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def _plot_pps_rtc_metric(self, keys, labels, ylabel, name, x_unit='time',
                             binwidth = 0.5, save=True, save_format=None,
                             dpi=None):
        """Plot PPS RTC metric

        Args:
            keys        : keys of the target metrics within the dataset
            labels      : labels for the plot legend
            ylabel      : y axis label
            name        : target filename
            x_unit      : Ynit for vs. time plot: 'time' in minutes or 'samples'
            binwidth    : Target histogram bin width
            save        : Save the figures
            save_format : Image format: 'png' or 'eps'

        """
        if (x_unit == "time"):
            t_start      = self.data[0]["t1"]
            x_axis_label = 'Time (min)'
            time_vec     = np.array([float(r["t1"] - t_start) for r in \
                                     self.data]) / NS_PER_MIN
        elif (x_unit == "samples"):
            x_axis_label = 'Realization'

        plt.figure(figsize=self.figsize)
        for key, label in zip(keys, labels):
            if (x_unit == "time"):
                x_vec = [time_vec[i] for i, r in enumerate(self.data) \
                           if key in r]
            elif (x_unit == "samples"):
                x_vec = [r["idx"] for r in self.data if key in r]
            y_vec = np.array([r[key] for r in self.data if key in r])
            plt.scatter(x_vec, y_vec, s = 1.0, label=label)

        plt.xlabel(x_axis_label)
        plt.ylabel(ylabel)
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + name + "_vs_time."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

        # Histogram
        plt.figure(figsize=self.figsize)
        for key,label in zip(keys, labels):
            y_vec = np.array([r[key] for r in self.data if key in r])
            if (len(y_vec) > 0):
                bins  = np.arange(np.floor(y_vec.min()),
                                  np.ceil(y_vec.max()) + binwidth, binwidth)
                plt.hist(y_vec, bins=bins, density=True, label=label)

        plt.xlabel(ylabel)
        plt.ylabel('Probability Density')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend()

        if (save):
            img_name   = self.prefix + name + "_hist."
            img_format = save_format or self.save_format
            img_dpi    = dpi or self.dpi
            plt.tight_layout()
            plt.savefig(self.path + img_name + img_format,
                        format=img_format, dpi=img_dpi)
        else:
            plt.show()
        plt.close()

    def plot_pps_err(self, x_unit='time', binwidth=0.5, save=True,
                     save_format=None, dpi=None):
        """Plot PPS synchronization error vs time and histogram

        Args:
            x_unit      : Unit for vs. time plot: 'time' in minutes or 'samples'
            binwidth    : Target histogram bin width
            save        : Save the figures
            save_format : Image format: 'png' or 'eps'

        """
        logger.info("Plot PPS sync error")
        keys   = ["pps_err", "pps_err2"]
        labels = ["RRU1", "RRU2"]
        ylabel = "PPS Sync Error (ns)"
        name   = "pps_err"
        self._plot_pps_rtc_metric(keys, labels, ylabel, name, x_unit, binwidth,
                                  save, save_format, dpi)

    def plot_pps_rtc_foffset_est(self, x_unit='time', binwidth=0.5, save=True,
                                 save_format=None, dpi=None):
        """Plot frequency offset estimates according to the PPS RTC

        These are equivalent to the output of the PI controller that is used for
        PPS synchronization.

        Args:
            x_unit      : Unit for vs. time plot: 'time' in minutes or 'samples'
            binwidth    : Target histogram bin width
            save        : Save the figures
            save_format : Image format: 'png' or 'eps'

        """
        logger.info("Plot frequency offset estimates of the PPS RTC")
        keys   = ["y_pps", "y_pps2"]
        labels = ["RRU1", "RRU2"]
        ylabel = "PPS frequency offset (ppb)"
        name   = "pps_foffset"
        self._plot_pps_rtc_metric(keys, labels, ylabel, name, x_unit, binwidth,
                                  save, save_format, dpi)

    def plot_error_vs_window(self, plot_info=False, save=True, save_format=None,
                             dpi=None):
        """Plot error vs window"""

        if (self.cache):
            window_cfg = self.cache.load('window')
        else:
            logger.warning("Unable to find cached file with window"
                            "optimization parameters")
            return

        for estimator in window_cfg.keys():

            est_key      = window_cfg[estimator]['est_key']
            est_name     = window_cfg[estimator]['name']
            N_best       = window_cfg[estimator]['N_best']
            error_metric = window_cfg[estimator]['error_metric']
            win_len      = window_cfg[estimator]['window_len']
            win_error    = window_cfg[estimator]['window_error']

            plt.figure(figsize=self.figsize)
            plt.scatter(win_len, win_error, s=3.0)
            plt.title(est_name)
            plt.xlabel("window length (samples)")
            plt.ylabel(f"{error_metric} (ns)")
            plt.grid(color='k', linewidth=.5, linestyle=':')

            if (plot_info):
                plt.text(0.99, 0.98, f"Best window length: {N_best:d}",
                         transform=plt.gca().transAxes, va='top', ha='right')

            if (save):
                img_name   = self.prefix + "win_opt_" + \
                             f"{est_key}_{error_metric}_error_vs_window."
                img_format = save_format or self.save_format
                img_dpi    = dpi or self.dpi
                plt.tight_layout()
                plt.savefig(self.path + img_name + img_format,
                            format=img_format, dpi=img_dpi)
            else:
                plt.show()
            plt.close()

