"""Analyse the LS performance as a function of window length
"""
import ptp.runner
import ptp.ls
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

# Run PTP simulation
n_iter = 10000
runner = ptp.runner.Runner(n_iter = n_iter)
runner.run()

# LS window lengths to evaluate:
window_len = np.arange(10, int(n_iter/10), 10)

# Compute max|TE| for each window length
max_te     = np.zeros(window_len.shape)
for i,N in enumerate(window_len):
    ls = ptp.ls.Ls(N, runner.data)
    ls.process(impl="eff")

    # Get LS time offset estimation errors
    x_err = np.array([r["x_ls_eff"] - r["x"] for r in runner.data
                      if "x_ls_eff" in r])

    # Erase LS results from runner data
    for r in runner.data:
        r.pop("x_ls_eff", None)

    # Save max|TE|
    max_te[i] = np.amax(np.abs(x_err))

plt.figure()
plt.scatter(window_len, max_te)
plt.xlabel('LS window length (samples)')
plt.ylabel("max|TE| (ns)")
plt.legend()
plt.savefig("plots/mtie_vs_window")

i_best = np.argmin(max_te)
print("Best evaluated window length: %d" %(window_len[i_best]))
