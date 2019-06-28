import ptp.runner
import ptp.ls
import ptp.metrics
import ptp.pktselection
import ptp.kalman
import ptp.frequency

# Parameters
n_iter     = 4096               # Number of runner iterations
N_ls       = 2048               # LS window
N_movavg   = 128                # Moving average window
N_median   = 128                # Sample-median window
N_min      = 128                # Sample-minimum window
N_ewma     = 64                 # EWMA window
freq_delta = 16                 # Freq. offset estimation delta

# Run PTP simulation
runner = ptp.runner.Runner(n_iter = n_iter, gamma_scale=1000)
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
pkts.process("min", ls_impl="eff")

# Exponentially weighted moving average
pkts.set_window_len(N_ewma)
pkts.process("ewma")

# Sample-mode
pkts.set_window_len(N_min)
pkts.process("mode")
pkts.process("mode", ls_impl="eff")

# Kalman (add frequency offset estimations to feed the Kalman filter)
freq_estimator = ptp.frequency.Estimator(runner.data, delta=freq_delta)
kalman         = ptp.kalman.Kalman(runner.data, runner.sync_period)
freq_estimator.process()
kalman.process()

# PTP analyser
analyser = ptp.metrics.Analyser(runner.data)
analyser.plot_toffset_vs_time()
analyser.plot_toffset_err_vs_time(show_raw = False)
analyser.plot_foffset_vs_time()
analyser.plot_mtie(show_raw = False)
analyser.plot_max_te(show_raw=False,
                     window_len = int((1/runner.sync_period) * 20))
analyser.plot_delay_hist()
