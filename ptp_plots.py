#!/usr/bin/env python
import ptp.simulation
import ptp.metrics

# Run PTP simulation
n_iter = 1000
simulation = ptp.simulation.Simulation(n_iter=n_iter)
simulation.run()

# Instantiate PTP analyser
analyser = ptp.metrics.Analyser(simulation.data)

# Demonstrate plots
analyser.plot_toffset_vs_time(show_best=True, save=True)
analyser.plot_foffset_vs_time(save=True)
analyser.plot_delay_hist(save=True)
analyser.plot_delay_vs_time(save=True)
analyser.plot_delay_est_err_vs_time(save=True)
analyser.plot_pdv_vs_time(save=True)
analyser.plot_mtie(save=True, show_ls=False)
