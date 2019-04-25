#!/usr/bin/env python

"""PTP Simulator

"""
import argparse, logging, sys, math, heapq
import ptp.runner

def main():
    parser = argparse.ArgumentParser(description="PTP Simulator")
    parser.add_argument('-N', '--num-iter',
                        default=10,
                        type=int,
                        help='Number of iterations.')
    parser.add_argument('-t', '--sim-step',
                        default=1e-9,
                        type=float,
                        help='Simulation time step in seconds.')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level")
    args     = parser.parse_args()
    args     = parser.parse_args()
    num_iter = args.num_iter
    sim_step = args.sim_step
    verbose  = args.verbose

    logging_level = 70 - (10 * verbose) if verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    runner = ptp.runner.Runner(n_iter=num_iter, sim_t_step=sim_step)
    runner.run()

if __name__ == "__main__":
    main()
