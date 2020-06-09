#!/usr/bin/env python

"""PTP Simulator

"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging, sys, math, heapq
import ptp.runner


def main():
    parser = ArgumentParser(description="PTP Simulator",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-N', '--num-iter',
                        default=10,
                        type=int,
                        help='Number of iterations.')
    parser.add_argument('-t', '--sim-step',
                        default=1e-9,
                        type=float,
                        help='Simulation time step in seconds.')
    parser.add_argument('-s', '--save',
                        default=False,
                        action='store_true',
                        help='Save dataset generated by the PTP runner.')
    parser.add_argument('-f', '--file',
                        default=None,
                        help='File containing simulation data to load.')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level")

    sim_group = parser.add_argument_group('Runner parameters')
    sim_group.add_argument('--rtc-clk-freq',  default=125e6,
                           type=float,
                           help="RTC's driving clock frequency in Hz")
    sim_group.add_argument('--rtc-resolution',  default=0,
                           type=float,
                           help="Fixed-point representation resolution in ns \
                           of the RTC's increment value")
    sim_group.add_argument('--sync-rate',  default=16,
                           type=float,
                           help="Sync transmission rate in packets per second")
    sim_group.add_argument('--freq-tolerance',  default=60,
                           type=float,
                           help="Slave RTC's frequency tolerance in ppb")
    sim_group.add_argument('--freq-rw', default=1e-18,
                           type=float,
                           help="Normalized variance of the slave RTC's \
                           random-walk in frequency (set 0 to disable this \
                           source of phase noise)")
    sim_group.add_argument('--phase-rw', default=1e-12,
                           type=float,
                           help="Normalized variance of the slave RTC's \
                           random-walk in phase, a.k.a. white noise in \
                           frequency (set 0 to disable this source of phase \
                           noise)")
    sim_group.add_argument('--pdv-distr',  default='Gamma',
                           type=str,
                           choices=["Gamma", "Gaussian"],
                           help="Distribution of the PDV experienced by PTP \
                           messages in the simulation (Gamma or Gaussian)")
    sim_group.add_argument('--gamma-shape',  default=None,
                           type=int,
                           help="Shape parameter of the Gamma distribution")
    sim_group.add_argument('--gamma-scale',  default=None,
                           type=int,
                           help="Scale parameter of the Gamma distribution")
    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    runner = ptp.runner.Runner(n_iter=args.num_iter, sim_t_step=args.sim_step,
                               sync_period=(1/args.sync_rate),
                               rtc_clk_freq=args.rtc_clk_freq,
                               rtc_resolution=args.rtc_resolution,
                               freq_tolerance=args.freq_tolerance,
                               freq_rw=args.freq_rw,
                               phase_rw=args.phase_rw,
                               pdv_distr=args.pdv_distr,
                               gamma_shape=args.gamma_shape,
                               gamma_scale=args.gamma_scale
                               )

    if (args.file is not None):
        runner.load(args.file)
        runner.dump()
    else:
        runner.run()

    if (args.save):
        runner.save()


if __name__ == "__main__":
    main()
