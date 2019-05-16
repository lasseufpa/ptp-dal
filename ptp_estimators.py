import ptp.runner
import ptp.ls
import ptp.metrics
import ptp.pktselection
import ptp.kalman
import ptp.frequency

# Parameters
n_iter   = 2000
N_ls     = 200 # Approximately best window length for LS
N_movavg = 30
N_median = 30
N_min    = 20
N_ewma   = 30

# Run PTP simulation
runner = ptp.runner.Runner(n_iter = n_iter)
runner.run()

# Least-squares estimator
ls = ptp.ls.Ls(N_ls, runner.data, runner.sync_period*1e9)
ls.process(impl="eff")

# Moving average
pkts = ptp.pktselection.PktSelection(N_movavg, runner.data)
pkts.process("average", avg_impl="recursive")

# Sample-median
pkts.set_window_len(N_median)
pkts.process("median")

# Sample-minimum
pkts.set_window_len(N_min)
pkts.process("min")

# Exponentially weighted moving average
pkts.set_window_len(N_ewma)
pkts.process("ewma")

# Kalman (add frequency offset estimations to feed the Kalman filter)
freq_estimator = ptp.frequency.Estimator(runner.data, period_ns=0)
kalman         = ptp.kalman.Kalman(runner.data, runner.sync_period)
freq_estimator.process()
kalman.process()

# PTP analyser
analyser = ptp.metrics.Analyser(runner.data)
analyser.plot_toffset_vs_time(show_ls=True, show_pkts =True, show_best=True, save=True)
analyser.plot_toffset_err_vs_time(show_raw=False, show_ls=True, show_pkts=True, save=True)
analyser.plot_foffset_vs_time(show_ls=True, show_kf=True, save=True)
analyser.plot_mtie(show_raw=False, show_ls=True, show_pkts=True, show_kf=True,
                   save=True)
