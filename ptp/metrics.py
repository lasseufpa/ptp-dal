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

    def plot_toffset_vs_time(self, show_best=False, show_ls=False, save=False):
        """Plot time offset vs Time

        A comparison between the measured time offset and the true time offset.

        Args:
            show_best : Enable to highlight the best measurements.
            show_ls   : Show least-squares fit
            save      : Save the figure

        """
        n_data  = len(self.data)
        x_tilde = np.zeros(n_data)
        x       = np.zeros(n_data)
        for idx, results in enumerate(self.data):
            x_tilde[idx] = results["x_est"]
            x[idx]       = results["x"]

        plt.figure()
        plt.scatter(range(0, n_data), x_tilde, label="Measurements", s = 1.0)
        plt.scatter(range(0, n_data), x, label="True Values", s = 1.0)

        # Least-squares estimations
        if (show_ls):
            idx_x_ls = list()
            x_ls     = list()
            for res in self.data:
                if ("x_ls" in res):
                    idx_x_ls.append(res["idx"])
                    x_ls.append(res["x_ls"])

            plt.scatter(idx_x_ls, x_ls,
                        label="LS Estimations", marker="x", s=50)

        # Best raw measurements
        if (show_best):
            err      = x_tilde - x
            best_idx = np.squeeze(np.where(abs(err) < 10))
            plt.scatter(best_idx, x_tilde[best_idx],
                        label="Accurate Measurements", s=50)

        plt.xlabel('Realization')
        plt.ylabel('Time offset')
        plt.legend()

        if (save):
            plt.savefig("plots/toffset_vs_time")
        else:
            plt.show()

    def plot_delay_vs_time(self, save=False):
        """Plot delay estimations vs time

        Args:
            save      : Save the figure

        """
        n_data = len(self.data)
        d      = np.zeros(n_data)
        d_est  = np.zeros(n_data)
        for idx, results in enumerate(self.data):
            d[idx]     = results["d"]
            d_est[idx] = results["d_est"]

        plt.figure()
        plt.scatter(range(0, n_data), d_est, label="Measurements", s = 1.0)
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
        n_data = len(self.data)
        d      = np.zeros(n_data)
        d_est  = np.zeros(n_data)
        for idx, results in enumerate(self.data):
            d[idx]     = results["d"]
            d_est[idx] = results["d_est"]

        plt.figure()
        plt.scatter(range(0, n_data), d_est -d, s = 1.0)
        plt.xlabel('Realization')
        plt.ylabel('Delay Estimation Error (ns)')

        if (save):
            plt.savefig("plots/delay_est_err_vs_time")
        else:
            plt.show()

    def plot_foffset_vs_time(self, show_ls=False, save=False):
        """Plot freq. offset vs time

        Args:
            show_ls   : Show least-squares estimations
            save      : Save the figure

        """
        n_data = len(self.data)
        y      = np.zeros(n_data)
        for idx, results in enumerate(self.data):
            y[idx] = results["rtc_y"]

        plt.figure()
        plt.scatter(range(0, n_data), y, label="True Values", s = 1.0)

        # Least-squares estimations
        if (show_ls):
            idx_y_ls = list()
            y_ls     = list()
            for res in self.data:
                if ("y_ls" in res):
                    idx_y_ls.append(res["idx"])
                    y_ls.append(res["y_ls"])

            plt.scatter(idx_y_ls, y_ls,
                        label="LS Estimations", marker="x", s=50)

        plt.xlabel('Realization')
        plt.ylabel('Frequency Offset (ppb)')
        plt.legend()

        if (save):
            plt.savefig("plots/foffset_vs_time")
        else:
            plt.show()
