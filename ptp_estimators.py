#!/usr/bin/env python
import ptp.simulation
import ptp.ls
import ptp.metrics
import ptp.pktselection
import ptp.kalman
import ptp.frequency

# Parameters
n_iter = 4096  # Number of simulation iterations
N_ls = 2048  # LS window
N_movavg = 128  # Moving average window
N_median = 128  # Sample-median window
N_min = 128  # Sample-minimum window
N_ewma = 64  # EWMA window
freq_delta = 16  # Freq. offset estimation delta

# Run PTP simulation
simulation = ptp.simulation.Simulation(n_iter=n_iter, gamma_scale=1000)
simulation.run()

# Run frequency estimations
freq_estimator = ptp.frequency.Estimator(simulation.data)
damping, loopbw = freq_estimator.optimize_loop()
freq_estimator.loop(damping=damping, loopbw=loopbw)

# Least-squares estimator
ls = ptp.ls.Ls(N_ls, simulation.data, simulation.sync_period * 1e9)
ls.process(impl="eff")

# Moving average
pkts = ptp.pktselection.PktSelection(N_movavg, simulation.data)
pkts.process("avg")

# Sample-median
pkts.set_window_len(N_median)
pkts.process("median")

# Sample-minimum
pkts.set_window_len(N_min)
pkts.process("min")

# Exponentially weighted moving average
pkts.set_window_len(N_ewma)
pkts.process("ewma")

# Sample-mode
pkts.set_window_len(N_min)
pkts.process("mode")

# Kalman (add frequency offset estimations to feed the Kalman filter)
freq_estimator = ptp.frequency.Estimator(simulation.data, delta=freq_delta)
kalman = ptp.kalman.Kalman(simulation.data, simulation.sync_period)
freq_estimator.process()
kalman.process()

# PTP analyser
analyser = ptp.metrics.Analyser(simulation.data)
analyser.plot_toffset_vs_time()
analyser.plot_toffset_err_vs_time(show_raw=False)
analyser.plot_foffset_vs_time()
analyser.plot_mtie(show_raw=False)
analyser.plot_max_te(show_raw=False,
                     window_len=int((1 / simulation.sync_period) * 20))
analyser.plot_delay_hist()
