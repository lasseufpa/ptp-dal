#!/usr/bin/env python

"""Acquisition of testbed data via serial

"""
import logging, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from pprint import pprint
import ptp.serial
from pyroe.roe import roe


def main():
    parser = ArgumentParser(description="Capture timestamps from FPGA",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rru',
                        default="rru_uart",
                        choices=["rru_uart", "rru2_uart"],
                        help='Target char device for UART communication with \
                        an RRU FPGA')
    parser.add_argument('--rru2',
                        default="rru2_uart",
                        choices=["rru_uart", "rru2_uart"],
                        help='Target char device for UART communication with \
                        an RRU2 FPGA')
    parser.add_argument('--bbu',
                        default="bbu_uart",
                        help='Target char device for UART communication \
                        with the BBU FPGA')
    parser.add_argument('--sensor',
                        default="roe_sensor",
                        help='Target char device for UART communication \
                        with the (Arduino) device that hosts the \
                        temperature sensor')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations. If set to 0, the \
                        acquisition will run indefinitely')
    parser.add_argument('-y', '--yes',
                        default=False,
                        action='store_true',
                        help='Default to answering yes on user prompting')
    parser.add_argument('-b', '--baudrate',
                        type=int,
                        default=230400,
                        help='UART baud rate for communication with RoE devices')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level.")
    roe.add_to_parser(parser) # add RoE options
    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # Parse RoE information
    roe_config, fh_config = roe.parse(args)

    # Dictionary containing the metadata
    bbu_ts_lat   = args.bbu_tstamp_latency
    rru1_ts_lat  = args.rru1_tstamp_latency
    rru2_ts_lat  = args.rru2_tstamp_latency
    metadata = {
        "oscillator" : args.oscillator,
        "sync_period": 1.0/args.sync_rate,
        "fh_traffic" : fh_config,
        "hops" : {
            "rru1" : args.rru1_hops,
            "rru2" : args.rru2_hops
        },
        "departure_ctrl" : (not args.no_departure_ctrl),
        "departure_gap"  : args.departure_gap,
        "tstamp_latency_corr" : {
            "bbu": {
                "tx": bbu_ts_lat[0],
                "rx": bbu_ts_lat[0] if (len(bbu_ts_lat) == 1) \
                      else bbu_ts_lat[1]
            },
            "rru1": {
                "tx": rru1_ts_lat[0],
                "rx": rru1_ts_lat[0] if (len(rru1_ts_lat) == 1) \
                      else rru1_ts_lat[1]
            },
            "rru2": {
                "tx": rru2_ts_lat[0],
                "rx": rru2_ts_lat[0] if (len(rru2_ts_lat) == 1) \
                      else rru2_ts_lat[1]
            },
        },
        "n_rru_ptp"    : args.n_rru_ptp,
        "pipelines"    : None if roe_config is None else roe_config['pipeline'],
        "start_time"   : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print("Metadata:")
    pprint(metadata)
    if (not args.yes):
        raw_resp = input("Proceed? [Y/n] ") or "Y"
        response = raw_resp.lower()

    if (args.yes or (response.lower() == "y")):
        serial = ptp.serial.Serial(args.rru, args.rru2, args.bbu, args.sensor,
                                   args.num_iter, metadata, roe_config,
                                   args.baudrate, yes=args.yes)
        serial.run()


if __name__ == "__main__":
    main()


