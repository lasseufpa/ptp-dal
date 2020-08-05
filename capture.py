#!/usr/bin/env python

"""Acquisition of timestamp data from the testbed
"""
import logging, sys, os, subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from pprint import pprint
import ptp.testbed
import pyroe.roe.opts
from pyroe.roe.roe import RoE


def _assert_free_to_capture():
    """Verify whether we are free to capture from FPGAs

    Throw exception in case there is another acquisition running.

    """
    our_pid = os.getpid()
    res     = subprocess.check_output(["pgrep", "-f", "capture.py", "-a"])

    for line in res.splitlines():
        pid = int(line.decode().split()[0])
        if (pid != our_pid):
            raise RuntimeError("An acquisition is already running on PID "
                               f"{pid}")


def _check_git_submodules():
    """Check if Git submodules are up-to-date"""

    res = subprocess.check_output(["git", "submodule", "status"])

    outdated = False
    for line in res.decode().splitlines():
        if ("+" in line):
            submodule = line.split()[1]
            logging.warning(f"{submodule} is not up-to-date")
            outdated = True

    if (outdated):
        raise RuntimeError("Git submodules are not up-to-date. "
                           "Run \"git submodule update\"")


def main():
    roe_parser = pyroe.roe.opts.get_parser()
    parser = ArgumentParser(description="Capture timestamps from FPGA",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[roe_parser],
                            conflict_handler='resolve')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations. If set to 0, the \
                        acquisition will run indefinitely')
    parser.add_argument('-y', '--yes',
                        default=False,
                        action='store_true',
                        help='Default to answering yes on user prompting')
    parser.add_argument('--dirty',
                        default=False,
                        action='store_true',
                        help='Allow execution with git submodules in dirty \
                        state')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level.")
    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # Check if there is another acquisition running
    _assert_free_to_capture()

    # Check if submodules are up-to-date
    if (not args.dirty):
        _check_git_submodules()

    # Initialize RoE system object
    roe_system = RoE(args)

    # Generate RoE metadata
    metadata = roe_system.manager.get_config()

    print("Metadata:")
    pprint(metadata)
    if (not args.yes):
        raw_resp = input("Proceed? [Y/n] ") or "Y"
        response = raw_resp.lower()

    if (not args.yes and (response != "y")):
        return

    # Program and configure the RoE device
    roe_system.config()

    # Start data acquisition
    acquisition = ptp.testbed.Acquisition(args.num_iter, roe_system, args.yes)
    acquisition.run()


if __name__ == "__main__":
    main()


