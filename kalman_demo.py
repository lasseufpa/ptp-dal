#!/usr/bin/env python

"""Experiment with Kalman Filter

Compare Kalman to LS

Suggested experiments:

- Set pdv_distr to "Gaussian" in order to experiment with Gaussian-distributed
  time offset noise.

- Change the freq_rw = 1e-18 to control how fast the frequency offset changes
  over time.

- Change the LS window length as it substantially affects the LS result.

- Change the transition and observation covariance matrices within the Kalman
  implementation.

- Change the delta of the frequency estimator to see how it affects Kalman
  performance.

"""
import ptp.simulation
import ptp.ls
import ptp.kalman
import ptp.metrics
import ptp.frequency


# Run PTP simulation
n_iter = 1e3
simulation = ptp.simulation.Simulation(n_iter = n_iter, pdv_distr="Gamma",
                                       freq_rw = 1e-18)
simulation.run()

# Least-squares estimator
N  = 128
ls = ptp.ls.Ls(N, simulation.data)
ls.process()

# Raw frequency estimations (differentiation of raw time offset measurements)
freq_estimator = ptp.frequency.Estimator(simulation.data, delta=1)
freq_estimator.process()

# Kalman
kalman = ptp.kalman.Kalman(simulation.data, simulation.sync_period)
kalman.process()

# PTP analyser
analyser = ptp.metrics.Analyser(simulation.data)
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
                   show_raw=False,
                   save=True)
