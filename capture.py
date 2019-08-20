#!/usr/bin/env python

"""Acquisition of testbed data via serial

"""
import argparse, configparser, logging, sys
from datetime import datetime
from pprint import pprint
import ptp.serial


DEFAULT_CONFIG = "config/capture.cfg"


def main():
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(DEFAULT_CONFIG)

    parser = argparse.ArgumentParser(description="Capture timestamps from FPGA")
    parser.add_argument('-t', '--target',
                        default="rru_uart",
                        choices=["bbu_uart", "rru_uart", "rru2_uart"],
                        help='Target char device for UART communication with \
                        FPGA (default: rru_uart).')
    parser.add_argument('--sensor',
                        default="roe_sensor",
                        help='Target char device for UART communication \
                        with the (Arduino) device that hosts the \
                        temperature sensor (default: roe_sensor).')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations. If set to 0, the \
                        acquisition will run indefinitely (default: 0).')
    parser.add_argument('-p', '--print-all', default=False, action='store_true',
                        help='Print out all non-timestamp logs from the FPGA')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level.")
    parser.add_argument('--oscillator',
                        default="xo",
                        choices=["ocxo", "xo"],
                        help='Define the oscillator type')
    parser.add_argument('--sync-period',
                        default=0.25,
                        type=float,
                        help='Sync transmission period in seconds')
    parser.add_argument('--hops',
                        type=int,
                        default=4,
                        help='Number of hops')

    fh_traffic_group = parser.add_argument_group('background traffic')
    fh_traffic_group.add_argument('--fh-traffic',
                                  default=False,
                                  action='store_true',
                                  help='Whether or not FH traffic is active')
    fh_traffic_group.add_argument('--type',
                                  choices=["inline","cross"],
                                  default=cfg_parser.get('FH-TRAFFIC',
                                                         'type'),
                                  help='Fronthaul traffic type')
    fh_traffic_group.add_argument('--fs',
                                  type=float,
                                  choices=[7680000, 30720000],
                                  default=cfg_parser.get('FH-TRAFFIC',
                                                         'fs'),
                                  help='LTE sample rate')
    fh_traffic_group.add_argument('--iq-size',
                                  type=int,
                                  choices=list(range(4,34,2)),
                                  default=cfg_parser.get('FH-TRAFFIC',
                                                         'iq_size'),
                                  help='IQ samples size')
    fh_traffic_group.add_argument('--n-spf',
                                  type=int,
                                  default=cfg_parser.get('FH-TRAFFIC',
                                                         'n_spf'),
                                  help='Number of IQ samples per frame')
    fh_traffic_group.add_argument('--n-rru-cfg',
                                  type=int,
                                  default=cfg_parser.get('FH-TRAFFIC',
                                                         'n_rru_cfg'),
                                  help='Number of RRUs that the BBU is \
                                  configured to deliver data to in DL')
    fh_traffic_group.add_argument('--n-rru-active',
                                  type=int,
                                  default=cfg_parser.get('FH-TRAFFIC',
                                                         'n_rru_active'),
                                  help='Number of RRUs actually active in the \
                                  testbed (delivering UL data) ')

    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # If background traffic is active, create a dictionary with all
    # information to save as metadata
    if (args.fh_traffic) :
        # Compute theoretical DL/UL bitrates
        eth_hdr_len    = 14*8
        fh_hdr_len     = 12*8
        n_axc_per_rru  = 2
        fh_payload_len = args.n_spf * args.iq_size
        fh_frame_len   = eth_hdr_len + fh_hdr_len + fh_payload_len
        i_bg           = args.n_spf / (n_axc_per_rru * args.fs);
        rate_per_rru   = fh_frame_len / i_bg
        bitrate_dl     = rate_per_rru * args.n_rru_cfg
        bitrate_ul     = rate_per_rru * args.n_rru_active

        fh_traffic = {
            "type" : args.type,
            "fs"   : args.fs, # in Hz
            "bitrate" : { # in bps
                "dl" : bitrate_dl,
                "ul" : bitrate_ul,
            },
            "iq_size" : args.iq_size,
            "n_spf" : args.n_spf,
            "n_rru_cfg" : args.n_rru_cfg,
            "n_rru_active" : args.n_rru_active,
        }
    else:
        fh_traffic = None

    # Dictionary containing the metadata
    metadata = {
        "oscillator": args.oscillator,
        "sync_period": args.sync_period,
        "fh_traffic" : fh_traffic,
        "hops" : args.hops,
        "start_time" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print("Metadata:")
    pprint(metadata)
    raw_resp = input("Proceed? [Y/n] ") or "Y"
    response = raw_resp.lower()

    if (response.lower() == "y"):
        serial = ptp.serial.Serial(args.target, args.sensor, args.num_iter,
                                   metadata)
        serial.run(args.print_all)

if __name__ == "__main__":
    main()
