import ptp.runner
import ptp.ls
import ptp.metrics
import ptp.pktselection

# Run PTP simulation
n_iter = 1000
runner = ptp.runner.Runner(n_iter = n_iter)
runner.run()

# Least-squares estimator
ls_N  = 200 # Approximately best window length for LS
ls = ptp.ls.Ls(ls_N, runner.data)
ls.process()
ls.process(impl="t1")
ls.process(impl="eff")

# Packet Selection estimator
pkts_N = 30 # Approximately best window length for PSA
pkts = ptp.pktselection.PktSelection(pkts_N, runner.data)
pkts.process("average", avg_impl="recursive")
pkts.process("ewma")
pkts.process("median")
pkts.process("min")

# PTP analyser
analyser = ptp.metrics.Analyser(runner.data)
analyser.plot_toffset_vs_time(show_ls=True, show_pkts =True, show_best=True, save=True)
analyser.plot_toffset_err_vs_time(show_raw=False, show_ls=True, show_pkts=True, save=True)
analyser.plot_foffset_vs_time(show_ls=True, save=True)
analyser.plot_mtie(show_raw=False, show_ls=True, show_pkts=True, save=True)
