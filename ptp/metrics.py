"""PTP metrics
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


pkts_keys   = ["pkts_average",
               "pkts_ewma",
               "pkts_median",
               "pkts_min",
               "pkts_max",
               "pkts_mode"]
pkts_labels = ["Sample-average",
               "EWMA",
               "Sample-median",
               "EAPF",
               "Sample-max",
               "Sample-mode"]
ls_keys     = ["ls_t2",
               "ls_t1",
               "ls_eff"]
ls_labels   = ["LSE (t2)",
               "LSE (t1)",
               "LSE"]


class Analyser():
    def __init__(self, data):
        """PTP metrics analyser

        Args:
            data : Array of objects with simulation data
        """
        self.data = data

    def mtie(self, tie):
        """Maximum time interval error (MTIE)

        Computes the MTIE based on time interval error (TIE) samples. The MTIE
        computes the peak-to-peak TIE over windows of increasing duration.

        Args:
            tie : Vector of TE values

        Returns:
            tau_array  : Observation intervals
            mtie_array : The calculated MTIE for each observation interval

        """
        window_inc    = 20 # Enlarge window by this amount on every interation
        n_samples     = len(tie) # total number of samples
        window_size   = 10
        mtie_array    = [0]
        tau_array     = [0]

        # Try until the window occupies half of the data length
        while (window_size <= n_samples/2):
            n_windows       = n_samples - window_size + 1
            mtie_candidates = np.zeros(n_windows)
            i_window        = 0
            i_start         = 0
            i_end           = window_size
            # Sweep overlapping windows with the current size:
            # NOTE: to speed up, not all possible overlapping windows are
            # evaluated. This is controlled by how much "i_start" increments
            # every time below.
            while (i_start < n_samples):
                tie_w = tie[i_start:i_end]     # current TE window
                # Get the MTIE candidate
                mtie_candidates[i_window] = np.amax(tie_w) - np.amin(tie_w)
                # Update indexes
                i_window += 1
                i_start  += 20
                i_end     = i_start + window_size

            # Final MTIE is the maximum among all candidates
            mtie = np.amax(mtie_candidates)

            # Save MTIE and its corresponding window duration
            mtie_array.append(mtie)
            tau_array.append(window_size)

            # Increase window size
            window_size += window_inc

        return tau_array, mtie_array

    def plot_toffset_vs_time(self, show_raw=True, show_best=False,
                             show_ls=False, show_pkts=False, show_kf=False,
                             show_true=True, n_skip_kf=0, save=False):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw  : Show raw measurements
            show_best : Enable to highlight the best measurements.
            show_ls   : Show least-squares fit
            show_pkts : Show Packet Selection fit
            show_kf   : Show Kalman filtering results
            n_skip_kf : Number of initial Kalman filter samples to skip
            show_true : Show true values
            save      : Save the figure

        """
        n_data  = len(self.data)

        plt.figure()

        if (show_raw):
            idx     = [r["idx"] for r in self.data]
            x_tilde = np.array([r["x_est"] for r in self.data])
            plt.scatter(idx, x_tilde,
                        label="Raw Measurements", s = 1.0)

        if (show_true):
            idx = [r["idx"] for r in self.data]
            x   = np.array([r["x"] for r in self.data])
            plt.scatter(idx, x, label="True Values", s = 1.0)

        # Least-squares estimations
        if (show_ls):
            for i, suffix in enumerate(ls_keys):
                key  = "x_" + suffix
                i_ls = [r["idx"] for r in self.data if key in r]
                x_ls = [r[key] for r in self.data if key in r]

                if (len(x_ls) > 0):
                    plt.scatter(i_ls, x_ls,
                                label=ls_labels[i], marker="x", s=1.0)

        # Kalman filtering output
        if (show_kf):
            i_kf  = [r["idx"] for r in self.data if "x_kf" in r]
            x_kf  = [r["x_kf"] for r in self.data if "x_kf" in r]
            if (n_skip_kf > 0):
                skip_label = " (after first %d)" %(n_skip_kf)
            else:
                skip_label = ""
            plt.scatter(i_kf[n_skip_kf:], x_kf[n_skip_kf:],
                        label="Kalman" + skip_label, marker="d", s=1.0)

        # Packet Selection estimation
        if (show_pkts):
            for i, suffix in enumerate(pkts_keys):
                key    = "x_" + suffix
                i_pkts = [r["idx"] for r in self.data if key in r]
                x_pkts = [r[key] for r in self.data if key in r]

                if (len(x_pkts) > 0):
                    plt.scatter(i_pkts, x_pkts,
                                label=pkts_labels[i], marker="v", s=1.0)

        # Best raw measurements
        if (show_best):
            assert(show_true), "show_best requires show_true"
            err      = x_tilde - x
            best_idx = np.squeeze(np.where(abs(err) < 10))
            plt.scatter(best_idx, x_tilde[best_idx],
                        label="Accurate Measurements", s=50)

        plt.xlabel('Realization')
        plt.ylabel('Time offset (ns)')
        plt.legend()

        if (save):
            plt.savefig("plots/toffset_vs_time", dpi=300)
        else:
            plt.show()

    def plot_toffset_err_vs_time(self, show_raw=True, show_ls=False,
                                 show_pkts=False, show_kf=False, save=False):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw  : Show raw measurements
            show_ls   : Show least-squares fit
            show_pkts : Show packet selection fit
            show_kf   : Show Kalman filtering results
            save      : Save the figure

        """
        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]
        n_data         = len(post_tran_data)

        plt.figure()

        if (show_raw):
            # Error of raw measurements
            x_tilde_err = [r["x_est"] - r["x"] for r in post_tran_data]
            plt.scatter(range(0, n_data), x_tilde_err,
                        label="Raw Measurements", s = 20.0, alpha=0.7)

        # Least-squares estimations
        if (show_ls):
            for i, suffix in enumerate(ls_keys):
                key  = "x_" + suffix
                i_ls = [r["idx"] for r in post_tran_data if key in r]
                x_ls = [r[key] - r["x"] for r in post_tran_data if key in r]

                if (len(x_ls) > 0):
                    plt.scatter(i_ls, x_ls,
                                label=ls_labels[i], marker="x", s=20.0, alpha=0.7)

        # Kalman filtering output
        if (show_kf):
            i_kf     = [r["idx"] for r in post_tran_data if "x_kf" in r]
            x_err_kf = [r["x_kf"] - r["x"] for r in post_tran_data if "x_kf" in r]
            plt.scatter(i_kf, x_err_kf,
                        label="Kalman", marker="d", s=20.0, alpha=0.7)

        # Packet Selection estimations
        if (show_pkts):
            for i, suffix in enumerate(pkts_keys):
                key    = "x_" + suffix
                i_pkts = [r["idx"] for r in post_tran_data if key in r]
                x_pkts = [r[key] - r["x"] for r in post_tran_data if key in r]

                if (len(x_pkts) > 0):
                    plt.scatter(i_pkts, x_pkts,
                                label=pkts_labels[i], marker="v", s=20.0, alpha=0.7)

        plt.xlabel('Realization')
        plt.ylabel('Time offset Error (ns)')
        plt.legend()

        if (save):
            plt.savefig("plots/toffset_err_vs_time", dpi=300)
        else:
            plt.show()

    def plot_delay_hist(self, save=False):
        """Plot delay histogram

        Args:
            save      : Save the figure

        """

        n_data = len(self.data)
        # Compute delays in microseconds
        d      = np.array([r["d"] for r in self.data]) / 1e3
        d_est  = np.array([r['d_est'] for r in self.data]) / 1e3

        plt.figure()
        plt.hist(d_est, bins=50, density=True, alpha=0.5,
                 label="Two-way Measurements")
        plt.hist(d, bins=50, density=True, alpha=0.5,
                 label="True Values")
        plt.xlabel('Delay (us)')
        plt.ylabel('Probability')
        plt.legend()

        if (save):
            plt.savefig("plots/delay_hist", dpi=300)
        else:
            plt.show()

    def plot_delay_vs_time(self, save=False):
        """Plot delay estimations vs time

        Args:
            save      : Save the figure

        """
        n_data = len(self.data)
        d      = [r["d"] for r in self.data]
        d_est  = [r['d_est'] for r in self.data]

        plt.figure()
        plt.scatter(range(0, n_data), d_est, label="Raw Measurements", s = 1.0)
        plt.scatter(range(0, n_data), d, label="True Values", s = 1.0)
        plt.xlabel('Realization')
        plt.ylabel('Delay Estimation (ns)')
        plt.legend()

        if (save):
            plt.savefig("plots/delay_vs_time", dpi=300)
        else:
            plt.show()

    def plot_delay_est_err_vs_time(self, save=False):
        """Plot delay estimations error vs time

        Args:
            save      : Save the figure

        """
        n_data    = len(self.data)
        d_est_err = [r["d_est"] - r["d"] for r in self.data]

        plt.figure()
        plt.scatter(range(0, n_data), d_est_err, s = 1.0)
        plt.xlabel('Realization')
        plt.ylabel('Delay Estimation Error (ns)')

        if (save):
            plt.savefig("plots/delay_est_err_vs_time", dpi=300)
        else:
            plt.show()

    def plot_foffset_vs_time(self, show_raw=True, show_ls=False, show_kf=False,
                             show_true=True, n_skip_kf=0, save=False):
        """Plot freq. offset vs time

        Args:
            show_raw  : Show raw measurements
            show_ls   : Show least-squares estimations
            show_kf   : Show Kalman filtering results
            n_skip_kf : Number of initial Kalman filter samples to skip
            show_true : Show true values
            save      : Save the figure

        """

        # To facilitate inspection, it is better to skip the transitory
        # (e.g. due to Kalman)
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]
        n_data         = len(post_tran_data)

        plt.figure()

        if (show_true):
            idx = [r["idx"] for r in post_tran_data]
            y   = [r["rtc_y"] for r in post_tran_data]
            plt.scatter(idx, y, label="True Values", s = 1.0)

        if (show_raw):
            i_y_tilde = [r["idx"] for r in post_tran_data if "y_est" in r]
            y_tilde   = [1e9*r["y_est"] for r in post_tran_data if "y_est" in r]
            plt.scatter(i_y_tilde, y_tilde, label="Raw Measurements", s = 2.0)

        # Show least-squares estimations
        if (show_ls):
            for i, suffix in enumerate(ls_keys):
                key  = "y_" + suffix
                i_ls = [r["idx"] for r in post_tran_data if key in r]
                y_ls = [1e9*r[key] for r in post_tran_data if key in r]

                if (len(y_ls) > 0):
                    plt.scatter(i_ls, y_ls,
                                label=ls_labels[i], marker="x", s=1.0)

        # Kalman filtering output
        if (show_kf):
            i_kf  = [r["idx"] for r in post_tran_data if "y_kf" in r]
            y_kf  = [1e9*r["y_kf"] for r in post_tran_data if "y_kf" in r]

            if (n_skip_kf > 0):
                skip_label = " (after first %d)" %(n_skip_kf)
            else:
                skip_label = ""

            plt.scatter(i_kf[n_skip_kf:], y_kf[n_skip_kf:],
                        label="Kalman" + skip_label, s=1.0)

        plt.xlabel('Realization')
        plt.ylabel('Frequency Offset (ppb)')
        plt.legend()

        if (save):
            plt.savefig("plots/foffset_vs_time", dpi=300)
        else:
            plt.show()

    def plot_pdv_vs_time(self, save=False):
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
            plt.savefig("plots/pdv_vs_time", dpi=300)
        else:
            plt.show()

    def plot_mtie(self, show_raw=True, show_ls=False, show_pkts=False,
                  show_kf=False, save=False):
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
            show_raw  : Show raw measurements
            show_ls   : Show least-squares fit
            show_pkts : Show Packet Selection fit
            show_kf   : Show Kalman filtering results
            save      : Save the figure

        """
        plt.figure()

        # Some methods such as Kalman, EWMA and the recursive moving average
        # have transitories. To analyse them, it is better to skip the
        # transitory by throwing away an arbitrary number of initial values. For
        # the sake of fairness, analyse all other methods also from this
        # transitory-removed dataset.
        n_skip         = int(0.2*len(self.data))
        post_tran_data = self.data[n_skip:]

        if (show_raw):
            # MTIE over raw time offset measurements
            x_err_raw         = [r["x_est_err"] for r in post_tran_data]
            i_raw             = [r["idx"] for r in post_tran_data ]
            tau_raw, mtie_raw = self.mtie(x_err_raw)

            plt.scatter(tau_raw, mtie_raw, label = "Raw Measurements", marker="x")

        # Least-squares estimations
        if (show_ls):
            for i, suffix in enumerate(ls_keys):
                key  = "x_" + suffix
                i_ls = [r["idx"] for r in post_tran_data if key in r]
                x_ls = [r[key] - r["x"] for r in post_tran_data if key in r]

                if (len(x_ls) > 0):
                    tau_ls, mtie_ls = self.mtie(x_ls)
                    plt.scatter(tau_ls, mtie_ls,
                            label=ls_labels[i], marker="x", s=80.0, alpha=1)

        # Packet Selection estimations
        if (show_pkts):
            for i, suffix in enumerate(pkts_keys):
                key    = "x_" + suffix
                i_pkts = [r["idx"] for r in post_tran_data if key in r]
                x_pkts = [r[key] - r["x"] for r in post_tran_data if key in r]

                if (len(x_pkts) > 0):
                    tau_pkts, mtie_pkts = self.mtie(x_pkts)
                    plt.scatter(tau_pkts, mtie_pkts,
                                label=pkts_labels[i], marker="v", s=80.0,
                                alpha=0.7)

        # Kalman filtering output
        if (show_kf):
            i_kf            = [r["idx"] for r in post_tran_data if "y_kf" in r]
            x_err_kf        = [r["x_kf"] - r["x"] for r in post_tran_data
                               if "x_kf" in r]
            tau_kf, mtie_kf = self.mtie(x_err_kf)
            plt.scatter(tau_kf, mtie_kf,
                        label="Kalman", marker="d", s=80.0, alpha=0.5)

        plt.xlabel ('Observation interval (samples)')
        plt.ylabel("MTIE (ns)")
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend(loc=0)

        if (save):
            plt.savefig("plots/mtie_vs_tau", dpi=300)
        else:
            plt.show()

