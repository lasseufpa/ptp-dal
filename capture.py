#!/usr/bin/env python

"""Acquisition of testbed data via serial

"""
import argparse, configparser, logging, sys
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
                        default="ocxo",
                        choices=["ocxo", "xo"],
                        help='Define the oscillator type')
    parser.add_argument('--sync-period',
                        default=0.25,
                        type=float,
                        help='Sync transmission period in seconds')

    bg_traffic_group = parser.add_argument_group('background traffic')
    bg_traffic_group.add_argument('--bg',
                                  default=False,
                                  action='store_true',
                                  help='Whether or not background traffic is \
                                  active')
    bg_traffic_group.add_argument('--type',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'type'),
                                  help='Fronthaul traffic type')
    bg_traffic_group.add_argument('--bitrate-dl',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'bitrate_dl'),
                                  help='Bitrate on downlink in Mbps')
    bg_traffic_group.add_argument('--bitrate-up',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'bitrate_up'),
                                  help='Bitrate on uplink in Mbps')
    bg_traffic_group.add_argument('--iq-size',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'iq_size'),
                                  help='IQ samples size')
    bg_traffic_group.add_argument('--n-spf',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'n_spf'),
                                  help='Number of IQ samples per frame')
    bg_traffic_group.add_argument('--n-rru-cfg',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'n_rru_cfg'),
                                  help='Number of RRUs to deliver')
    bg_traffic_group.add_argument('--n-rru-active',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'n_rru_active'),
                                  help='Number of active RRUs')
    bg_traffic_group.add_argument('--topology',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'topology'),
                                  help='Type of network topology')
    bg_traffic_group.add_argument('--hops',
                                  default=cfg_parser.get('BG-TRAFFIC',
                                                         'hops'),
                                  help='Number of hops')

    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # If background traffic is active, create a dictionary with all
    # information to save as metadata
    if (args.bg) :
        bg_traffic = {
            "type" : args.type,
            "bitrate" : {
                "dl" : args.bitrate_dl,
                "ul" : args.bitrate_up,
            },
            "iq_size" : args.iq_size,
            "n_spf" : args.n_spf,
            "n_rru_cfg" : args.n_rru_cfg,
            "n_rru_active" : args.n_rru_active,
            "topology" : args.topology,
            "hops" : args.hops
        }
    else:
        bg_traffic = None

    # Dictionary containing the metadata
    metadata = {
        "oscillator": args.oscillator,
        "sync_period": args.sync_period,
        "bg_traffic" : bg_traffic
    }

    serial = ptp.serial.Serial(args.target, args.sensor, args.num_iter, metadata)
    serial.run(args.print_all)

if __name__ == "__main__":
    main()
