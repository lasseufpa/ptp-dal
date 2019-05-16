"""PTP metrics
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


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
            # Sweep all possible windows with the current size:
            while ((window_size + i_window) <= n_samples):
                i_end = window_size + i_window
                tie_w = tie[i_window:i_end]     # current TE window
                # Get the MTIE candidate
                mtie_candidates[i_window] = np.amax(tie_w) - np.amin(tie_w)
                i_window  += 1

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
        x_tilde = np.array([r["x_est"] for r in self.data])

        plt.figure()

        if (show_raw):
            plt.scatter(range(0, n_data), x_tilde,
                        label="Raw Measurements", s = 1.0)

        if (show_true):
            x       = np.array([r["x"] for r in self.data])
            plt.scatter(range(0, n_data), x, label="True Values", s = 1.0)

        # Least-squares estimations
        if (show_ls):
            i_ls_t2  = [r["idx"] for r in self.data if "x_ls_t2" in r]
            i_ls_t1  = [r["idx"] for r in self.data if "x_ls_t1" in r]
            i_ls_eff = [r["idx"] for r in self.data if "x_ls_eff" in r]
            x_ls_t2  = [r["x_ls_t2"] for r in self.data if "x_ls_t2" in r]
            x_ls_t1  = [r["x_ls_t1"] for r in self.data if "x_ls_t1" in r]
            x_ls_eff = [r["x_ls_eff"] for r in self.data if "x_ls_eff" in r]

            if (len(x_ls_t1) > 0):
                plt.scatter(i_ls_t1, x_ls_t1,
                            label="LSE (t1)", marker="x", s=1.0)
            if (len(x_ls_t2) > 0):
                plt.scatter(i_ls_t2, x_ls_t2,
                            label="LSE (t2)", marker="x", s=1.0)
            if (len(x_ls_eff) > 0):
                plt.scatter(i_ls_eff, x_ls_eff,
                            label="LSE", marker="x", s=1.0)

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
            i_pkt_avg     = [r["idx"] for r in self.data if "x_pkts_average" in r]
            i_pkt_ewma    = [r["idx"] for r in self.data if "x_pkts_ewma" in r]
            i_pkt_median  = [r["idx"] for r in self.data if "x_pkts_median" in r]
            i_pkt_minimum = [r["idx"] for r in self.data if "x_pkts_min" in r]
            x_pkt_avg     = [r["x_pkts_average"] for r in self.data if
                             "x_pkts_average" in r]
            x_pkt_ewma    = [r["x_pkts_ewma"] for r in self.data if
                             "x_pkts_ewma" in r]
            x_pkt_median  = [r["x_pkts_median"] for r in self.data if
                             "x_pkts_median" in r]
            x_pkt_minimum = [r["x_pkts_min"] for r in self.data if "x_pkts_min" in r]

            if (len(x_pkt_avg) > 0):
                plt.scatter(i_pkt_avg, x_pkt_avg,
                            label="Sample-average", marker="v", s=1.0)

            if (len(x_pkt_ewma) > 0):
                plt.scatter(i_pkt_ewma, x_pkt_ewma,
                            label="EWMA", marker="v", s=1.0)

            if (len(x_pkt_median) > 0):
                plt.scatter(i_pkt_median, x_pkt_median,
                            label="Sample-median", marker="v", s=1.0)

            if (len(x_pkt_minimum) > 0):
                plt.scatter(i_pkt_minimum, x_pkt_minimum,
                            label="Sample-minimum", marker="v", s=1.0)

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
            plt.savefig("plots/toffset_vs_time")
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
        n_data      = len(self.data)

        plt.figure()

        if (show_raw):
            # Error of raw measurements
            x_tilde_err = [r["x_est"] - r["x"] for r in self.data]
            plt.scatter(range(0, n_data), x_tilde_err,
                        label="Raw Measurements", s = 1.0)

        # Least-squares estimations
        if (show_ls):
            i_ls_t2      = [r["idx"] for r in self.data if "x_ls_t2" in r]
            i_ls_t1      = [r["idx"] for r in self.data if "x_ls_t1" in r]
            i_ls_eff     = [r["idx"] for r in self.data if "x_ls_eff" in r]
            x_ls_err_t2  = [r["x_ls_t2"] - r["x"] for r in self.data
                            if "x_ls_t2" in r]
            x_ls_err_t1  = [r["x_ls_t1"] - r["x"] for r in self.data
                            if "x_ls_t1" in r]
            x_ls_err_eff = [r["x_ls_eff"] - r["x"] for r in self.data
                            if "x_ls_eff" in r]

            if (len(x_ls_err_t1) > 0):
                plt.scatter(i_ls_t1, x_ls_err_t1,
                            label="LSE (t1)", marker="x", s=1.0)
            if (len(x_ls_err_t2) > 0):
                plt.scatter(i_ls_t2, x_ls_err_t2,
                            label="LSE (t2)", marker="x", s=1.0)
            if (len(x_ls_err_eff) > 0):
                plt.scatter(i_ls_eff, x_ls_err_eff,
                            label="LSE", marker="x", s=1.0)

        # Kalman filtering output
        if (show_kf):
            i_kf     = [r["idx"] for r in self.data if "x_kf" in r]
            x_err_kf = [r["x_kf"] - r["x"] for r in self.data if "x_kf" in r]
            plt.scatter(i_kf, x_err_kf,
                        label="Kalman", marker="d", s=1.0)

        # Packet Selection estimations
        if (show_pkts):
            i_pkt_avg         = [r["idx"] for r in self.data if "x_pkts_average" in r]
            i_pkt_ewma        = [r["idx"] for r in self.data if "x_pkts_ewma" in r]
            i_pkt_median      = [r["idx"] for r in self.data if "x_pkts_median" in r]
            i_pkt_minimum     = [r["idx"] for r in self.data if "x_pkts_min" in r]
            x_pkt_err_avg     = [r["x_pkts_average"] - r["x"] for r in
                                 self.data if "x_pkts_average" in r]
            x_pkt_err_ewma    = [r["x_pkts_ewma"] - r["x"] for r in self.data
                                 if "x_pkts_ewma" in r]
            x_pkt_err_median  = [r["x_pkts_median"] - r["x"] for r in self.data
                                 if "x_pkts_median" in r]
            x_pkt_err_minimum = [r["x_pkts_min"] - r["x"] for r in self.data if
                                 "x_pkts_min" in r]

            if (len(x_pkt_err_avg) > 0):
                plt.scatter(i_pkt_avg, x_pkt_err_avg,
                            label="Sample-average", marker="v", s=1.0)

            if (len(x_pkt_err_ewma) > 0):
                plt.scatter(i_pkt_ewma, x_pkt_err_ewma,
                            label="EWMA", marker="v", s=1.0)

            if (len(x_pkt_err_median) > 0):
                plt.scatter(i_pkt_median, x_pkt_err_median,
                            label="Sample-median", marker="v", s=1.0)

            if (len(x_pkt_err_minimum) > 0):
                plt.scatter(i_pkt_minimum, x_pkt_err_minimum,
                            label="Sample-minimum", marker="v", s=1.0)

        plt.xlabel('Realization')
        plt.ylabel('Time offset Error (ns)')
        plt.legend()

        if (save):
            plt.savefig("plots/toffset_err_vs_time")
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
            plt.savefig("plots/delay_vs_time")
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
            plt.savefig("plots/delay_est_err_vs_time")
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
        n_data    = len(self.data)

        plt.figure()

        if (show_true):
            y = [r["rtc_y"] for r in self.data]
            plt.scatter(range(0, n_data), y, label="True Values", s = 1.0)

        if (show_raw):
            i_y_tilde = [r["idx"] for r in self.data if "y_est" in r]
            y_tilde   = [1e9*r["y_est"] for r in self.data if "y_est" in r]
            plt.scatter(i_y_tilde, y_tilde, label="Raw Measurements", s = 2.0)

        # Show least-squares estimations
        if (show_ls):
            i_ls_t2  = [r["idx"] for r in self.data if "y_ls_t2" in r]
            i_ls_t1  = [r["idx"] for r in self.data if "y_ls_t1" in r]
            i_ls_eff = [r["idx"] for r in self.data if "y_ls_eff" in r]
            y_ls_t2  = [1e9*r["y_ls_t2"] for r in self.data if "y_ls_t2" in r]
            y_ls_t1  = [1e9*r["y_ls_t1"] for r in self.data if "y_ls_t1" in r]
            y_ls_eff = [1e9*r["y_ls_eff"] for r in self.data if "y_ls_eff" in r]

            if (len(y_ls_t2) > 0):
                plt.scatter(i_ls_t2, y_ls_t2,
                            label="LSE (t2)", marker="x", s=1.0)
            if (len(y_ls_t1) > 0):
                plt.scatter(i_ls_t1, y_ls_t1,
                            label="LSE (t1)", marker="x", s=1.0)
            if (len(y_ls_eff) > 0):
                plt.scatter(i_ls_eff, y_ls_eff,
                            label="LSE", marker="x", s=1.0)

        # Kalman filtering output
        if (show_kf):
            i_kf  = [r["idx"] for r in self.data if "y_kf" in r]
            y_kf  = [1e9*r["y_kf"] for r in self.data if "y_kf" in r]

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
            plt.savefig("plots/foffset_vs_time")
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
            plt.savefig("plots/pdv_vs_time")
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

        if (show_raw):
            # MTIE over raw time offset measurements
            x_err_raw         = np.array([r["x_est_err"] for r in self.data ])
            i_raw             = [r["idx"] for r in self.data ]
            tau_raw, mtie_raw = self.mtie(x_err_raw)

            plt.scatter(tau_raw, mtie_raw, label = "Raw Measurements", marker="x")

        # Least-squares estimations
        if (show_ls):
            i_ls_t2      = [r["idx"] for r in self.data if "x_ls_t2" in r]
            i_ls_t1      = [r["idx"] for r in self.data if "x_ls_t1" in r]
            i_ls_eff     = [r["idx"] for r in self.data if "x_ls_eff" in r]
            x_ls_err_t2  = [r["x_ls_t2"] - r["x"] for r in self.data
                            if "x_ls_t2" in r]
            x_ls_err_t1  = [r["x_ls_t1"] - r["x"] for r in self.data
                            if "x_ls_t1" in r]
            x_ls_err_eff = [r["x_ls_eff"] - r["x"] for r in self.data
                            if "x_ls_eff" in r]

            if (len(x_ls_err_t1) > 0):
                tau_ls_t1, mtie_ls_t1 = self.mtie(x_ls_err_t1)
                plt.scatter(tau_ls_t1, mtie_ls_t1,
                            label="LSE (t1)", marker="x", s=80.0, alpha=1)
            if (len(x_ls_err_t2) > 0):
                tau_ls_t2, mtie_ls_t2 = self.mtie(x_ls_err_t2)
                plt.scatter(tau_ls_t2, mtie_ls_t2,
                            label="LSE (t2)", marker="x", s=80.0, alpha=1)
            if (len(x_ls_err_eff) > 0):
                tau_ls_eff, mtie_ls_eff = self.mtie(x_ls_err_eff)
                plt.scatter(tau_ls_eff, mtie_ls_eff,
                            label="LSE", marker="x", s=80.0, alpha=1)

        # Packet Selection estimations
        if (show_pkts):
            i_pkts_avg        = [r["idx"] for r in self.data if "x_pkts_average" in r]
            i_pkts_ewma       = [r["idx"] for r in self.data if "x_pkts_ewma" in r]
            i_pkts_median     = [r["idx"] for r in self.data if "x_pkts_median" in r]
            i_pkts_minimum    = [r["idx"] for r in self.data if "x_pkts_min" in r]
            x_pkts_err_avg    = [r["x_pkts_average"] - r["x"] for r in self.data
                            if"x_pkts_average" in r]
            x_pkts_err_ewma   = [r["x_pkts_ewma"] - r["x"] for r in self.data
                            if"x_pkts_ewma" in r]
            x_pkts_err_median = [r["x_pkts_median"] - r["x"] for r in self.data
                            if"x_pkts_median" in r]
            x_pkts_err_min    = [r["x_pkts_min"] - r["x"] for r in self.data
                            if "x_pkts_min" in r]

            if (len(x_pkts_err_avg) > 0):
                tau_pkts_avg, mtie_pkts_avg = self.mtie(x_pkts_err_avg)
                plt.scatter(tau_pkts_avg, mtie_pkts_avg,
                            label="Sample-average", marker="v", s=80.0, alpha=0.7)
            if (len(x_pkts_err_ewma) > 0):
                tau_pkts_ewma, mtie_pkts_ewma = self.mtie(x_pkts_err_ewma)
                plt.scatter(tau_pkts_ewma, mtie_pkts_ewma,
                            label="EWMA", marker="v", s=80.0, alpha=0.7)
            if (len(x_pkts_err_median) > 0):
                tau_pkts_median, mtie_pkts_median = self.mtie(x_pkts_err_median)
                plt.scatter(tau_pkts_median, mtie_pkts_median,
                            label="Sample-median", marker="v", s=80.0, alpha=0.7)
            if (len(x_pkts_err_min) > 0):
                tau_pkts_min, mtie_pkts_min = self.mtie(x_pkts_err_min)
                plt.scatter(tau_pkts_min, mtie_pkts_min,
                            label="Sample-minimum", marker="v", s=80.0, alpha=0.7)

        # Kalman filtering output
        if (show_kf):
            # Kalman has a transitory. Try to skip it by throwing away an
            # arbitrary number of initial values.
            kf_data         = self.data[200:]
            i_kf            = [r["idx"] for r in kf_data if "y_kf" in r]
            x_err_kf        = [r["x_kf"] - r["x"] for r in kf_data if "x_kf" in r]
            tau_kf, mtie_kf = self.mtie(x_err_kf)
            plt.scatter(tau_kf, mtie_kf,
                        label="Kalman", marker="d", s=80.0, alpha=0.5)

        plt.xlabel ('Observation interval (samples)')
        plt.ylabel("MTIE (ns)")
        plt.grid(color='k', linewidth=.5, linestyle=':')
        plt.legend(loc=0)

        if (save):
            plt.savefig("plots/mtie_vs_tau")
        else:
            plt.show()

