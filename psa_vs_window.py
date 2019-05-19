"""Analyse the Packet Selection performance as a function of window length
"""
import ptp.runner
import ptp.pktselection
import ptp.ls
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


# Parameters
n_iter   = 5000
strategy = "average"          # "average", "ewma", "median" or "min".
ls_impl  = None               # LS implementation to use on PSA
n_skip   = 300                # samples to skip due to transitory
min_len  = 16                 # Minimum window length to test
max_len  = 512                # Maximum window length to test

print("Strategy: %s" %(strategy))

# Run PTP simulation
runner = ptp.runner.Runner(n_iter = n_iter)
runner.run()

# PSA window lengths to evaluate:
window_len = np.arange(min_len, max_len)

# Compute max|TE| for each window length
max_te     = np.zeros(window_len.shape)

for i,N in enumerate(window_len):
    if(ls_impl):
        ls = ptp.ls.Ls(N, runner.data)
        ls.process(impl=ls_impl)

    pkts = ptp.pktselection.PktSelection(N, runner.data)
    pkts.process(strategy, ls_impl=ls_impl)

    # EWMA and the recursive moving average have transitories. Try to
    # skip it by throwing away an arbitrary number of initial values.
    post_tran_data = runner.data[n_skip:]

    # Get PSA time offset estimation errors
    x_err = np.array([r["x_pkts_{}".format(strategy)] - r["x"] for r
                      in post_tran_data
                      if "x_pkts_{}".format(strategy) in r])

    # Erase PSA results from runner data
    for r in runner.data:
        r.pop("x_pkts_{}".format(strategy), None)

        if (ls_impl):
            r.pop("x_ls_{}".format(ls_impl), None)

    # Save max|TE|
    max_te[i] = np.amax(np.abs(x_err))

plt.figure()
plt.scatter(window_len, max_te, s=1.0)
plt.xlabel('PSA window length (samples)')
plt.ylabel("max|TE| (ns)")
plt.savefig("plots/psa_vs_window")

i_best = np.argmin(max_te)
print("Best evaluated window length: %d" %(window_len[i_best]))
