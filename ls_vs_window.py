"""Analyse the LS performance as a function of window length
"""
import argparse
import ptp.runner, ptp.ls, ptp.reader
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="LS performance vs window")
    parser.add_argument('-f', '--file',
                        default=None,
                        help='Serial capture file.')
    args     = parser.parse_args()

    if (args.file is None):
        # Run PTP simulation
        n_iter = 20000
        ptp_src = ptp.runner.Runner(n_iter = n_iter)
        ptp_src.run()
    else:
        ptp_src = ptp.reader.Reader(args.file)
        ptp_src.process()

    # LS window lengths to evaluate:
    max_N      = int(len(ptp_src.data)/2)
    window_len = np.arange(8, 1024, 8)
    n_iter     = len(window_len)

    # Compute max|TE| for each window length
    max_te     = np.zeros(window_len.shape)
    last_print = 0
    for i,N in enumerate(window_len):

        progress = (i/n_iter)
        if (progress - last_print > 0.1):
            print("LS vs. window progress %5.2f%%" %(progress*100))
            last_print = progress

        ls = ptp.ls.Ls(N, ptp_src.data)
        ls.process(impl="eff")

        # Get LS time offset estimation errors
        x_err = np.array([r["x_ls_eff"] - r["x"] for r in ptp_src.data
                          if "x_ls_eff" in r])

        # Erase LS results from runner data
        for r in ptp_src.data:
            r.pop("x_ls_eff", None)

        # Save max|TE|
        max_te[i] = np.amax(np.abs(x_err))

    plt.figure()
    plt.scatter(window_len, max_te)
    plt.xlabel('LS window length (samples)')
    plt.ylabel("max|TE| (ns)")
    plt.legend()
    plt.savefig("plots/ls_max_te_vs_window")

    i_best = np.argmin(max_te)
    print("Best evaluated window length: %d" %(window_len[i_best]))


if __name__ == "__main__":
    main()

