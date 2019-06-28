"""PTP metrics
"""
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import re

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
            "pkts_min"    : {"label": "EAPF",
                             "marker": "v",
                             "show": True},
            "pkts_min_ls" : {"label": "EAPF with LS",
                             "marker": "v",
                             "show": True},
            "pkts_max"    : {"label": "Sample-max",
                             "marker": "v",
                             "show": True},
            "pkts_mode"   : {"label": "Sample-mode",
                             "marker": "v",
                             "show": True},
            "pkts_mode_ls": {"label": "Sample-mode with LS",
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
    def __init__(self, data):
        """PTP metrics analyser

        Args:
            data : Array of objects with simulation data
        """
        self.data = data

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

    def mtie(self, tie, window_step = 20, starting_window = 10):
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
        max_win_size = n_samples/2
        n_tau        = math.floor((max_win_size - starting_window)/window_step)\
                       + 1

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
            window_size = window_size + window_step

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
                    preffix_key = (re.search(r'(?<=show_).*', k)).group(0)
                    # Find the dict keys that match with the preffix_keys
                    key_values  = [key for key in est_keys if
                                   re.match(r'^{}_*'.format(preffix_key), key)]
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
                             n_skip_kf=0, x_unit='time', save=True,
                             save_format='png'):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw    : Show raw measurements
            show_best   : Enable to highlight the best measurements.
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            n_skip_kf   : Number of initial Kalman filter samples to skip
            show_true   : Show true values
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        n_data  = len(self.data)

        # Time axis
        t_start  = self.data[0]["t1"]
        time_vec = np.array([float(r["t1"] - t_start) for r in self.data])\
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

                x_est = [r[key] for r in self.data if key in r]

                # Define the x axis - either in time or in samples
                if (x_unit == "time"):
                    x_axis_vec   = [time_vec[i] for i, r in
                                    enumerate(self.data) if key in r]
                elif (x_unit == "samples"):
                    x_axis_vec   = [r["idx"] for r in self.data if key in r]

                if (len(x_est) > 0):
                    plt.scatter(x_axis_vec, x_est,
                                label=value["label"], marker=value["marker"],
                                s=1.0)

        # Best raw measurements
        if (show_best):
            x_tilde  = np.array([r["x_est"] for r in self.data])
            x        = np.array([r["x"] for r in self.data])

            # Find best raw measurements (with error under 10 ns)
            err      = x_tilde - x
            best_idx = np.squeeze(np.where(abs(err) < 10))

            # Define the x axis - either in time or in samples
            if (x_unit == "time"):
                x_axis_vec   = time_vec[best_idx]
            elif (x_unit == "samples"):
                x_axis_vec   = best_idx

            plt.scatter(x_axis_vec, x_tilde[best_idx],
                        label="Accurate Measurements", s=50)

        plt.xlabel(x_axis_label)
        plt.ylabel('Time offset (ns)')
        plt.legend()

        if (save):
            plt.savefig("plots/toffset_vs_time", format=save_format, dpi=300)
        else:
            plt.show()

    @dec_plot_filter
    def plot_toffset_err_vs_time(self, show_raw=True, show_ls=True,
                                 show_pkts=True, show_kf=True, x_unit='time',
                                 save=True, save_format='png'):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show packet selection fit
            show_kf     : Show Kalman filtering results
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]
        n_data         = len(post_tran_data)

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
                    x_axis_vec = [time_vec[i] for i, r in
                                  enumerate(post_tran_data) if key in r]
                elif (x_unit == "samples"):
                    x_axis_vec = [r["idx"] for r in post_tran_data if key in r]

                if (len(x_est) > 0):
                    plt.scatter(x_axis_vec, x_est,
                                label=value["label"], marker=value["marker"],
                                s=20.0, alpha=0.7)

        plt.xlabel(x_axis_label)
        plt.ylabel('Time offset Error (ns)')
        plt.legend()

        if (save):
            plt.savefig("plots/toffset_err_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()

    def plot_delay_hist(self, show_raw=True, show_true=True, n_bins=50,
                        save=True, save_format='png'):
        """Plot delay histogram

        Plot histogram of delays in microseconds.

        Args:
            show_raw    : Show histogram of raw delay measurements
            show_true   : Show histogram true delay values
            n_bins      : Target number of bins
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        n_data = len(self.data)

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

        plt.xlabel('Delay (us)')
        plt.ylabel('Probability Density')
        plt.legend()

        if (save):
            plt.savefig("plots/delay_hist", format=save_format, dpi=300)
        else:
            plt.show()

    def plot_delay_vs_time(self, save=True, save_format='png'):
        """Plot delay estimations vs time

        Args:
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        n_data = len(self.data)
        d      = [r["d"] for r in self.data]
        d_est  = [r["d_est"] for r in self.data]

        plt.figure()
        plt.scatter(range(0, n_data), d_est, label="Raw Measurements", s = 1.0)
        plt.scatter(range(0, n_data), d, label="True Values", s = 1.0)
        plt.xlabel('Realization')
        plt.ylabel('Delay Estimation (ns)')
        plt.legend()

        if (save):
            plt.savefig("plots/delay_vs_time", format=save_format, dpi=300)
        else:
            plt.show()

    def plot_delay_est_err_vs_time(self, save=True, save_format='png'):
        """Plot delay estimations error vs time

        Args:
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        n_data    = len(self.data)
        d_est_err = [r["d_est"] - r["d"] for r in self.data]

        plt.figure()
        plt.scatter(range(0, n_data), d_est_err, s = 1.0)
        plt.xlabel('Realization')
        plt.ylabel('Delay Estimation Error (ns)')

        if (save):
            plt.savefig("plots/delay_est_err_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()

    @dec_plot_filter
    def plot_foffset_vs_time(self, show_raw=True, show_ls=True, show_kf=True,
                             show_true=True, n_skip_kf=0, x_unit='time',
                             save=True, save_format='png'):
        """Plot freq. offset vs time

        Args:
            show_raw    : Show raw measurements
            show_ls     : Show least-squares estimations
            show_kf     : Show Kalman filtering results
            n_skip_kf   : Number of initial Kalman filter samples to skip
            show_true   : Show true values
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """

        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]
        n_data         = len(post_tran_data)

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
            plt.savefig("plots/foffset_vs_time", format=save_format,
                        dpi=300)
        else:
            plt.show()

    def plot_pdv_vs_time(self, save=True, save_format='png'):
        """Plot PDV over time

        Each value represents the measured difference of the current Sync delay
        with respect to the delay experienced by the previous Sync. Note that
        the actual delay is not measurable, but the difference in delay is. We
        define the PDV as follows:

        pdv = (t2[k] - t2[k-1]) - (t1[k] - t1[k-1])
        pdv = delta_t2[k] - delta_t1[k]

        If delta_t2[k] == delta_t1[k], it means both Sync messsages experienced
        the same delay, which we don't know.

        Args:
            save      : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
        n_data  = len(self.data)

        # Timestamps
        t2      = [res["t2"] for res in self.data]
        t1      = [res["t1"] for res in self.data]

        # Deltas
        delta_t1 = np.asarray([float(t1[i+1] - t) for i,t in enumerate(t1[:-1])])
        delta_t2 = np.asarray([float(t2[i+1] - t) for i,t in enumerate(t2[:-1])])

        # PDV
        pdv = delta_t2 - delta_t1

        plt.figure()
        plt.scatter(range(0, n_data-1), pdv, s = 1.0)
        plt.xlabel('Realization')
        plt.ylabel('Delay Variation (ns)')

        if (save):
            plt.savefig("plots/pdv_vs_time", format=save_format, dpi=300)
        else:
            plt.show()

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
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
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
                    plt.scatter(tau_est, mtie_est,
                                label=value["label"], marker=value["marker"],
                                s=80.0, alpha=0.7)

        plt.xlabel('Observation interval (samples)')
        plt.ylabel('MTIE (ns)')
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend(loc=0)

        if (save):
            plt.savefig("plots/mtie_vs_tau", format=save_format, dpi=300)
        else:
            plt.show()

    @dec_plot_filter
    def plot_max_te(self, window_len, show_raw=True, show_ls=True,
                    show_pkts=True, show_kf=True, x_unit='time', save=True,
                    save_format='png'):
        """Plot Max|TE| vs time.

        Args:
            window_len  : Window lengths
            show_raw    : Show raw measurements
            show_ls     : Show least-squares fit
            show_pkts   : Show Packet Selection fit
            show_kf     : Show Kalman filtering results
            x_unit      : Horizontal axis unit: 'time' in minutes or 'samples'
            save        : Save the figure
            save_format : Select image format: 'png' or 'eps'

        """
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
            plt.savefig("plots/max_te_vs_time", format=save_format, dpi=300)
        else:
            plt.show()
