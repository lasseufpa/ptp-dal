"""Experiment with Kalman Filter

Compare Kalman to LS

Suggested experiments:

- Set pdv_distr to "Gaussian" in order to experiment with Gaussian-distributed
  time offset noise.

- Change the rtc_stability = 0.1 to control how fast the frequency offset
  changes over time.

- Change the LS window length as it substantially affects the LS result.

- Change the transition and observation covariance matrices within the Kalman
  implementation.

"""
import ptp.runner
import ptp.ls
import ptp.kalman
import ptp.metrics

# Run PTP simulation
n_iter = 1e3
runner = ptp.runner.Runner(n_iter = n_iter, pdv_distr="Gamma",
                           rtc_stability = 0.1, freq_est_per = 0)
runner.run()

# Least-squares estimator
N  = 128
ls = ptp.ls.Ls(N, runner.data)
ls.process()

# Kalman
kalman = ptp.kalman.Kalman(runner.data, runner.sync_period)
kalman.process()

# PTP analyser
analyser = ptp.metrics.Analyser(runner.data)
analyser.plot_toffset_vs_time(show_ls=True,
                              show_kf=True,
                              save=True)
analyser.plot_toffset_err_vs_time(show_ls=True,
                                  show_kf=True,
                                  show_raw=False,
                                  save=True)
analyser.plot_foffset_vs_time(show_ls=True,
                              show_kf=True,
                              show_raw=False,
                              save=True)
analyser.plot_mtie(show_ls=True,
                   show_kf=True,
                   save=True)
