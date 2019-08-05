#!/usr/bin/env python

"""Acquisition of testbed data via serial

"""
import argparse, logging, sys
import ptp.serial

def main():
    parser = argparse.ArgumentParser(description="Capture timestamps from FPGA")
    parser.add_argument('-t', '--target',
                        default="rru_uart",
                        choices=["bbu_uart", "rru_uart", "rru2_uart"],
                        help='Target char device for UART communication with ' +
                        'FPGA (default: rru_uart).')
    parser.add_argument('--sensor',
                        default="roe_sensor",
                        help='Target char device for UART communication ' +
                        'with the (Arduino) device that hosts the ' +
                        'temperature sensor (default: roe_sensor).')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations. If set to 0, the ' +
                        ' acquisition will run indefinitely (default: 0).')
    parser.add_argument('-p', '--print-all', default=False, action='store_true',
                        help='Print out all non-timestamp logs from the FPGA')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level.")
    parser.add_argument('--oscillator',
                       default="ocxo",
                       choices=["ocxo", "xo"],
                       help='Define the oscillator type')
    parser.add_argument('--sync-period',
                      default=0.25,
                      type=float,
                      help='Sync transmission period in seconds')

    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # Dictionary containing the metadata
    metadata = {
        "oscillator": args.oscillator,
        "sync_period": args.sync_period
    }

    serial = ptp.serial.Serial(args.target, args.sensor, args.num_iter, metadata)
    serial.run(args.print_all)

if __name__ == "__main__":
    main()
