"""PTP metrics
"""
import math, logging, re, os, json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)


NS_PER_MIN = (60 * 1e9)
est_keys = {"raw"         : {"label": "Raw Measurements",
                             "marker": None,
                             "show": True},
            "true"        : {"label": "True Values",
                             "marker": None,
                             "show": True},
            "pkts_average": {"label": "Sample-average",
                             "marker": "v",
                             "show": True},
            "pkts_ewma"   : {"label": "EWMA",
                             "marker": "v",
                             "show": True},
            "pkts_median" : {"label": "Sample-median",
                             "marker": "v",
                             "show": True},
            "pkts_min"    : {"label": "Sample-min",
                             "marker": "v",
                             "show": True},
            "pkts_max"    : {"label": "Sample-max",
                             "marker": "v",
                             "show": True},
            "pkts_mode"   : {"label": "Sample-mode",
                             "marker": "v",
                             "show": True},
            "ls_t2"       : {"label": "LSE (t2)",
                             "marker": "x",
                             "show": True},
            "ls_t1"       : {"label": "LSE (t1)",
                             "marker": "x",
                             "show": True},
            "ls_eff"      : {"label": "LSE",
                             "marker": "x",
                             "show": True},
            "kf"          : {"label": "Kalman",
                             "marker": "d",
                             "show": True}}

class Analyser():
    def __init__(self, data, file=None):
        """PTP metrics analyser

        Args:
            data : Array of objects with simulation data
            file : Path of the file
        """
        self.data = data
        self.path = self._set_path(file)

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
            path     = 'plots/' + basename + '/'
        else:
            path = 'plots/'

        # Create the folder if it doesn't exist
        if not os.path.isdir(path):
            os.makedirs(path)

        return path

    def save_metadata(self, metadata):
        """Save metadata info on the path where plots are saved
        """

        with open(os.path.join(self.path, 'info.txt'), 'w') as outfile:
            json.dump(metadata, outfile)

    def ptp_exchanges_per_sec(self):
        """Compute average number of PTP exchanges per second

        Returns:
            The computed average

        """
        logger.info("Analyze PTP exchanges per second")
        start_time = self.data[0]["t1"]
        end_time   = self.data[-1]["t1"]
        duration   = float(end_time - start_time)
        n_per_sec  = 1e9 * len(self.data) / duration

        print("Average no. of PTP exchanges per second: %f" %(n_per_sec))

        return n_per_sec

    def delay_asymmetry(self, verbose=True):
        """Analyze the delay asymmetry

        Compute and print some relevant asymmetry metrics.

        Returns:
            Average delay asymmetry

        """
        logger.info("Analyze delay asymmetry")
        d_asym = np.array([r['asym'] for r in self.data])
        d_ms   = np.array([r["d"] for r in self.data])
        d_sm   = np.array([r["d_bw"] for r in self.data])

        if (verbose):
            print("Delay asymmetry analysis:\n")
            print("Metric \t%12s\t%12s\t%12s" %("m-to-s", "s-to-m", "asymmetry"))
            print("Average\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                np.mean(d_ms), np.mean(d_sm), np.mean(d_asym)))
            print("Minimum\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                np.amin(d_ms), np.amin(d_sm), (np.amin(d_ms) - np.amin(d_sm))/2))
            print("Maximum\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                np.amax(d_ms), np.amax(d_sm), (np.amax(d_ms) - np.amax(d_sm))/2))
            print("Median\t%9.2f ns\t%9.2f ns\t%9.2f ns" %(
                np.median(d_ms), np.median(d_sm),
                (np.median(d_ms) - np.median(d_sm))/2))

        return np.mean(d_asym)

    def toffset_err_stats(self):
        """Print the time offset estimation error statistics

        """
        # Skip the transitory (e.g. due to Kalman)
        logger.info("Eval time offset estimation error statistics")
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        for suffix, value in est_keys.items():
            key   = "x_est" if (suffix == "raw") else "x_" + suffix
            x_err = [r[key] - r["x"] for r in post_tran_data if key in r]

            if (len(x_err) > 0):
                print("[%14s] Mean: % 7.2f ns\tSdev: % 7.2f ns" %(
                    key, np.mean(x_err), np.std(x_err)))

            del x_err

    def foffset_err_stats(self):
        """Print the frequency offset estimation error statistics

        """
        # Skip the transitory (e.g. due to Kalman)
        logger.info("Eval frequency offset estimation error statistics")

        for suffix, value in est_keys.items():
            key   = "y_est" if (suffix == "raw") else "y_" + suffix
            y_err = [1e9*(r[key] - r["rtc_y"]) for r in self.data if key in r]

            if (len(y_err) > 0):
                print("[%14s] Mean: % 7.4f ppb\tSdev: % 7.4f ppb" %(
                    key, np.mean(y_err), np.std(y_err)))

            del y_err

    def rolling_window_mtx(self, x, window_size):
        """Compute all overlapping (rolling) observation windows in a matrix

        Args:
            x           : observation vector that is supposed to be split into
                          overlapping windows
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

        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def mtie(self, tie, window_step = 2, starting_window = 16):
        """Maximum time interval error (MTIE)

        Computes the MTIE based on time interval error (TIE) samples. The MTIE
        computes the peak-to-peak TIE over windows of increasing duration.

        Args:
            tie             : Vector of TIE values
            window_step     : Enlarge window by this step on every iteration
            starting_window : Starting window size

        Returns:
            tau_array  : MTIE observation intervals
            mtie_array : The calculated MTIE for each observation interval

        """
        n_samples   = len(tie) # total number of samples
        tie         = np.array(tie)

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
            parted_array = self.rolling_window_mtx(tie, window_size)

            # Get maximum and minimum of each window
            window_max = np.max(parted_array, axis = 1)
            window_min = np.min(parted_array, axis = 1)

            # MTIE candidates (for each window):
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
        a sliding window.

        Args:
            window_len = Window length
            te         = Vector of time error (TE) values

        Returns:
            max_te     = The calculated Max|TE| over a sliding window

        """
        n_data = len(te)
        max_te = np.zeros(n_data)

        for i in range(0, (n_data - window_len)):
            # Window start and end indexes
            i_s = i
            i_e = i + window_len

            # Max|TE| within the observation window
            max_te_w = np.amax(np.abs(te[i_s:i_e]))
            max_te[i_e - 1] = max_te_w

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
                             show_pkts=True, show_kf=True, show_true=True,
                             n_skip=None, x_unit='time', save=True,
                             save_format='png'):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw    : Show raw measurements
            show_best   : Enable to highlight the best measurements.
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            n_skip      : Number of initial samples to skip
            show_true   : Show true values
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        plt.figure()

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
        plt.legend()

        if (save):
            plt.savefig(self.path + "toffset_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_toffset_err_vs_time(self, show_raw=True, show_ls=True,
                                 show_pkts=True, show_kf=True, n_skip=None,
                                 x_unit='time', save=True, save_format='png'):
        """Plot time offset error vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show packet selection fit
            show_kf     : Show Kalman filtering results
            n_skip      : Number of initial samples to skip
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        plt.figure()

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
        plt.legend()

        if (save):
            plt.savefig(self.path + "toffset_err_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_toffset_err_hist(self, show_raw=True, show_ls=True, show_pkts=True,
                              show_kf=True, n_bins=50, save=True,
                              save_format='png'):
        """Plot time offset error histogram

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show packet selection fit
            show_kf     : Show Kalman filtering results
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot time offset estimation error histogram")
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        plt.figure()

        for suffix, value in est_keys.items():
            if (value["show"]):
                key   = "x_est" if (suffix == "raw") else "x_" + suffix
                x_err = [r[key] - r["x"] for r in post_tran_data if key in r]

                if (len(x_err) > 0):
                    plt.hist(x_err, bins=50, density=True, alpha=0.7,
                             label=value["label"])

        plt.xlabel('Time offset Error (ns)')
        plt.ylabel('Probability Density')
        plt.legend()

        if (save):
            plt.savefig(self.path + "toffset_err_hist", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_delay_hist(self, show_raw=True, show_true=True, n_bins=50,
                        split=False, save=True, save_format='png'):
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

        """
        logger.info("Plot delay histogram")
        x_label = 'Delay (us)'
        y_label = 'Probability Density'

        if (split):
            plots   = list()

            if (show_raw):
                d_est = np.array([r['d_est'] for r in self.data]) / 1e3
                plt.figure()
                plt.hist(d_est, bins=n_bins, density=True)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title("Two-way Measurements")
                plots.append({"plt" : plt.gcf(),
                              "label" : "raw"})

            if (show_true):
                d = np.array([r["d"] for r in self.data]) / 1e3
                d_bw = np.array([r["d_bw"] for r in self.data]) / 1e3

                plt.figure()
                plt.hist(d, bins=n_bins, density=True)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title("True master-to-slave")
                plots.append({"plt": plt.gcf(),
                              "label": "m2s"})

                plt.figure()
                plt.hist(d_bw, bins=n_bins, density=True)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title("True slave-to-master")
                plots.append({"plt": plt.gcf(),
                              "label": "s2m"})

            for p in plots:
                if (save):
                    p["plt"].savefig(self.path + "delay_hist_" + p["label"],
                                     format=save_format, dpi=300)
                else:
                    p["plt"].show()
        else:
            # Single plot

            plt.figure()
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
            plt.legend()

            if (save):
                plt.savefig(self.path + "delay_hist", format=save_format,
                            dpi=300)
            else:
                plt.show()
            plt.close()

    def plot_delay_vs_time(self, x_unit='time', split=False, save=True,
                           save_format='png'):
        """Plot delay estimations vs time

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            split       : Whether to split m-to-s and s-to-m plots
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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
        d_est  = [r["d_est"] for r in self.data]

        if (split):
            plt.figure()
            plt.scatter(x_axis_vec, d, s = 1.0)
            plt.xlabel(x_axis_label)
            plt.ylabel('m-to-s delay (ns)')

            if (save):
                plt.savefig(self.path + "m2s_delay_vs_time", format=save_format,
                            dpi=300)
            else:
                plt.show()
            plt.close()

            plt.figure()
            plt.scatter(x_axis_vec, d_bw, s = 1.0)
            plt.xlabel(x_axis_label)
            plt.ylabel('s-to-m delay (ns)')

            if (save):
                plt.savefig(self.path + "s2m_delay_vs_time", format=save_format,
                            dpi=300)
            else:
                plt.show()
            plt.close()
        else:
            plt.figure()
            plt.scatter(x_axis_vec, d_est, label="Raw Measurements", s = 1.0)
            plt.scatter(x_axis_vec, d, label="True Values", s = 1.0)
            plt.xlabel(x_axis_label)
            plt.ylabel('Delay Estimation (ns)')
            plt.legend()

            if (save):
                plt.savefig(self.path + "delay_vs_time", format=save_format,
                            dpi=300)
            else:
                plt.show()
            plt.close()

    def plot_delay_est_err_vs_time(self, x_unit='time', save=True, save_format='png'):
        """Plot delay estimations error vs time

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        plt.figure()
        plt.scatter(x_axis_vec, d_est_err, s = 1.0)
        plt.xlabel(x_axis_label)
        plt.ylabel('Delay Estimation Error (ns)')

        if (save):
            plt.savefig(self.path + "delay_est_err_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_delay_asym_hist(self, n_bins=50, save=True, save_format='png'):
        """Plot delay asymmetry histogram

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        logger.info("Plot delay asymmetry histogram")

        plt.figure()
        d_asym = np.array([r['asym'] for r in self.data]) / 1e3
        plt.hist(d_asym, bins=n_bins, density=True)
        plt.xlabel('Delay asymmetry (us)')
        plt.ylabel('Probability Density')

        if (save):
            plt.savefig(self.path + "delay_asym_hist", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_delay_asym_vs_time(self, save=True, x_unit='time',
                                save_format='png'):
        """Plot delay asymmetry over time

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        plt.figure()
        plt.scatter(time_vec, d_asym, s=1.0)
        plt.xlabel(x_axis_label)
        plt.ylabel('Delay asymmetry (us)')

        if (save):
            plt.savefig(self.path + "delay_asym_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_foffset_vs_time(self, show_raw=True, show_ls=True, show_kf=True,
                             show_true=True, n_skip=None, x_unit='time',
                             save=True, save_format='png'):
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

        plt.figure()

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
        plt.legend()

        if (save):
            plt.savefig(self.path + "foffset_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_foffset_err_vs_time(self, show_raw=True, show_ls=True,
                                 show_kf=True, n_skip_kf=0, x_unit='time',
                                 save=True, save_format='png'):
        """Plot freq. offset estimation error vs time

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares estimations
            show_kf     : Show Kalman filtering results
            n_skip_kf   : Number of initial Kalman filter samples to skip
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        plt.figure()

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
        plt.legend()

        if (save):
            plt.savefig(self.path + "foffset_err_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_foffset_err_hist(self, show_raw=True, show_ls=True, show_kf=True,
                              n_bins=50, save=True, save_format='png'):
        """Plot frequency offset error histogram

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_kf     : Show Kalman filtering results
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        logger.info("Plot time offset estimation error histogram")
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        plt.figure()

        for suffix, value in est_keys.items():
            if (value["show"]):
                key   = "y_est" if (suffix == "raw") else "y_" + suffix
                y_err = [r[key] - r["rtc_y"] for r in post_tran_data if key in r]

                if (len(y_err) > 0):
                    plt.hist(y_err, bins=50, density=True, alpha=0.7,
                             label=value["label"])

        plt.xlabel('Frequency Offset Error (ppb)')
        plt.ylabel('Probability Density')
        plt.legend()

        if (save):
            plt.savefig(self.path + "foffset_err_hist", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_pdv_vs_time(self, x_unit='time', save=True, save_format='png'):
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

        plt.figure()
        plt.scatter(x_axis_vec[1:], diff_t2_1, s = 1.0, label="m-to-s")
        plt.scatter(x_axis_vec[1:], diff_t4_3, s = 1.0, label="s-to-m")
        plt.xlabel(x_axis_label)
        plt.ylabel('Delay Variation (ns)')
        plt.legend()

        if (save):
            plt.savefig(self.path + "pdv_vs_time", format=save_format, dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_pdv_hist(self, n_bins=50, save=True, save_format='png'):
        """Plot PDV histogram

        See explanation of "plot_pdv_vs_time".

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        logger.info("Plot PDV histogram")

        # Timestamp differences
        t2_1 = np.array([float(r["t2"] - r["t1"]) for r in self.data])
        t4_3 = np.array([float(r["t4"] - r["t3"]) for r in self.data])

        # Diffs
        diff_t2_1 = np.diff(t2_1)
        diff_t4_3 = np.diff(t4_3)

        plt.figure()
        plt.hist(diff_t2_1, bins=n_bins, density=True, alpha=0.7,
                 label="m-to-s")
        plt.hist(diff_t4_3, bins=n_bins, density=True, alpha=0.7,
                 label="s-to-m")
        plt.xlabel('Delay Variation (ns)')
        plt.ylabel('Probability Density')
        plt.legend()

        if (save):
            plt.savefig(self.path + "pdv_hist", format=save_format, dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_toffset_diff_vs_time(self, x_unit='time', save=True,
                                  save_format='png'):
        """Plot time offset diff vs. time

        It is useful to analyze how x[n] varies between consecutive PTP
        exchanges. This plot shows (x[n] - x[n-1]) over time.

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        # True time offset
        x = np.array([r["x"] for r in self.data])

        plt.figure()
        plt.scatter(x_axis_vec[1:], np.diff(x), s = 1.0)
        plt.xlabel(x_axis_label)
        plt.ylabel('x[n] - x[n-1] (ns)')
        plt.title('Time offset diff')

        if (save):
            plt.savefig(self.path + "toffset_diff_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_toffset_diff_hist(self, n_bins=50, save=True, save_format='png'):
        """Plot time offset diff histogram

        Args:
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        logger.info("Plot PDV vs. time")

        # True time offset
        x = np.array([r["x"] for r in self.data])

        plt.figure()
        plt.hist(np.diff(x), bins=n_bins, density=True)
        plt.xlabel('x[n] - x[n-1] (ns)')
        plt.ylabel('Probability Density')
        plt.title('Time offset diff')

        if (save):
            plt.savefig(self.path + "toffset_diff_hist", format=save_format,
                        dpi=300)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_mtie(self, show_raw=True, show_ls=True, show_pkts=True,
                  show_kf=True, save=True, save_format='png'):
        """Plot MTIE versus the observation interval(Tau)

        Plots MTIE. The time interval error (TIE) samples are assumed to be
        equal to the time offset estimation errors. The underlying assumption is
        that in practice these estimations would be used to correct the clock
        and thus the resulting TIE with respect to the reference time would
        correspond to the error in the time offset estimation.

        The observation window durations of the associated time error samples
        are assumed to be given in terms of number of samples that they contain,
        rather than their actual time durations. This is not strictly how MTIE
        is computed, but useful for the evaluation and simpler to implement.

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            n_skip      : Number of initial samples to skip
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        logger.info("Plot MTIE")
        plt.figure()

        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        for suffix, value in est_keys.items():
            if (value["show"]):
                key   = "x_est" if (suffix == "raw") else "x_" + suffix
                x_est = [r[key] - r["x"] for r in post_tran_data if key in r]

                if (len(x_est) > 0):
                    tau_est, mtie_est = self.mtie(x_est)
                    plt.semilogx(tau_est, mtie_est,
                                 label=value["label"], marker=value["marker"],
                                 alpha=0.7, basex=2)

        plt.xlabel('Observation interval (samples)')
        plt.ylabel('MTIE (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend(loc=0)

        if (save):
            plt.savefig(self.path + "mtie_vs_tau", format=save_format, dpi=300)
        else:
            plt.show()
        plt.close()

    @dec_plot_filter
    def plot_max_te(self, window_len, show_raw=True, show_ls=True,
                    show_pkts=True, show_kf=True, n_skip=None, x_unit='time',
                    save=True, save_format='png'):
        """Plot Max|TE| vs time.

        Args:
            window_len  : Window lengths
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            n_skip      : Number of initial samples to skip
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        plt.figure()

        for suffix, value in est_keys.items():
            if (value["show"]):
                key   = "x_est" if (suffix == "raw") else "x_" + suffix
                x_est = [r[key] - r["x"] for r in post_tran_data if key in r]

                # Define the x axis - either in time or in samples
                if (x_unit == "time"):
                    x_axis_vec   = [time_vec[i] for i, r in
                                    enumerate(post_tran_data) if key in r]
                elif (x_unit == "samples"):
                    x_axis_vec   = [r["idx"] for r in post_tran_data if key in r]

                if (len(x_est) > 0):
                    max_te_est = self.max_te(x_est, window_len)
                    plt.plot(x_axis_vec, max_te_est,
                             label=value["label"], markersize=1)

        plt.xlabel(x_axis_label)
        plt.ylabel('Max|TE| (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend(loc=0)

        if (save):
            plt.savefig(self.path + "max_te_vs_time", format=save_format, dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_temperature(self, x_unit='time', save=True, save_format='png'):
        """Plot temperature vs time

        Args:
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

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

        temp = [r["temp"] for r in self.data if "temp" in r]

        plt.figure()
        plt.scatter(x_axis_vec, temp, s = 1.0)
        plt.xlabel(x_axis_label)
        plt.ylabel('Temperature (C)')

        if (save):
            plt.savefig(self.path + "temperature_vs_time", format=save_format, dpi=300)
        else:
            plt.show()
        plt.close()
