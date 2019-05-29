#!/usr/bin/env python

"""Acquisition of testbed data via serial

"""
import argparse, logging, sys
import ptp.serial

def main():
    parser = argparse.ArgumentParser(description="PTP log reader test")
    parser.add_argument('-t', '--target',
                        default="rru_uart",
                        choices=["bbu_uart", "rru_uart", "rru2_uart"],
                        help='Target UART connection.')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations.')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level")
    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    serial = ptp.serial.Serial(args.target, args.num_iter)
    serial.run()

if __name__ == "__main__":
    main()
