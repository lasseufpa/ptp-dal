import ptp.runner
import ptp.ls
import ptp.metrics

# Run PTP simulation
n_iter = 1000
runner = ptp.runner.Runner(n_iter = n_iter, target_ds = 'toffset_regression')
runner.run()

# Least-squares estimator
N  = 64
ls = ptp.ls.Ls(N, runner.data)
ls.process()

# PTP analyser
analyser = ptp.metrics.Analyser(runner.data)
analyser.plot_toffset_vs_time(show_ls=True, show_best=True, save=True)
analyser.plot_foffset_vs_time(show_ls=True, save=True)

