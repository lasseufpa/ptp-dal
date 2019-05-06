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

    def plot_toffset_vs_time(self, show_best=False, show_ls=False,
                             show_kf=False, save=False):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_best : Enable to highlight the best measurements.
            show_ls   : Show least-squares fit
            show_kf   : Show Kalman filtering results
            save      : Save the figure

        """
        n_data  = len(self.data)
        x_tilde = np.array([r["x_est"] for r in self.data])
        x       = np.array([r["x"] for r in self.data])

        plt.figure()
        plt.scatter(range(0, n_data), x_tilde, label="Raw Measurements", s = 1.0)
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
                            label="LS Estimations - t1", marker="x", s=1.0)
            if (len(x_ls_t2) > 0):
                plt.scatter(i_ls_t2, x_ls_t2,
                            label="LS Estimations - t2", marker="v", s=1.0)
            if (len(x_ls_eff) > 0):
                plt.scatter(i_ls_eff, x_ls_eff,
                            label="LS Estimations - eff", marker="d", s=1.0)

        # Kalman filtering output
        if (show_kf):
            i_kf  = [r["idx"] for r in self.data if "x_kf" in r]
            x_kf  = [r["x_kf"] for r in self.data if "x_kf" in r]
            plt.scatter(i_kf, x_kf,
                        label="Kalman", marker="v", s=1.0)

        # Best raw measurements
        if (show_best):
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
                                 show_kf=False, save=False):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_raw  : Show raw measurements
            show_ls   : Show least-squares fit
            show_kf   : Show Kalman filtering results
            save      : Save the figure

        """
        n_data      = len(self.data)

        plt.figure()

        if (show_raw):
            # Error of raw measurements
            x_tilde_err = [r["x_est"] - r["x"] for r in self.data]
            plt.scatter(range(0, n_data), x_tilde_err, label="Raw Measurements", s = 1.0)

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
                            label="LS Estimations - t1", marker="x", s=1.0)
            if (len(x_ls_err_t2) > 0):
                plt.scatter(i_ls_t2, x_ls_err_t2,
                            label="LS Estimations - t2", marker="v", s=1.0)
            if (len(x_ls_err_eff) > 0):
                plt.scatter(i_ls_eff, x_ls_err_eff,
                            label="LS Estimations - eff", marker="d", s=1.0)

        # Kalman filtering output
        if (show_kf):
            i_kf     = [r["idx"] for r in self.data if "x_kf" in r]
            x_err_kf = [r["x_kf"] - r["x"] for r in self.data if "x_kf" in r]
            plt.scatter(i_kf, x_err_kf,
                        label="Kalman", marker="v", s=1.0)

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
                             save=False):
        """Plot freq. offset vs time

        Args:
            show_raw  : Show raw measurements
            show_ls   : Show least-squares estimations
            show_kf   : Show Kalman filtering results
            save      : Save the figure

        """
        n_data    = len(self.data)
        y         = [r["rtc_y"] for r in self.data]

        plt.figure()
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
                            label="LS Estimations - t2", marker="x", s=1.0)
            if (len(y_ls_t1) > 0):
                plt.scatter(i_ls_t1, y_ls_t1,
                            label="LS Estimations - t1", marker="v", s=1.0)
            if (len(y_ls_eff) > 0):
                plt.scatter(i_ls_eff, y_ls_eff,
                            label="LS Estimations - t1 nominal", marker="d",
                            s=1.0)

        # Kalman filtering output
        if (show_kf):
            i_kf  = [r["idx"] for r in self.data if "y_kf" in r]
            y_kf  = [1e9*r["y_kf"] for r in self.data if "y_kf" in r]
            plt.scatter(i_kf, y_kf,
                        label="Kalman", s=1.0)

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

