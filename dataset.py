#!/usr/bin/env python3
import argparse, sys, logging, requests, json
from tabulate import tabulate
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ptp.datasets


def main():
    parser    = ArgumentParser(description="Manage datasets",
                               formatter_class=ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest='command')

    # Download datasets
    download = subparser.add_parser('download',
                                    help="Download datasets")
    download.add_argument('--file', '-f',
                          help='Dataset file name',
                          required=True)

    # Search datasets
    search = subparser.add_parser('search',
                                  help="Search for datasets using its metadata")
    search.add_argument('--name',
                        default=None,
                        help='Dataset name')
    search.add_argument('--oscillator',
                        choices=["ocxo", "xo"],
                        help='Define the oscillator type')
    search.add_argument('--sync-period',
                        type=float,
                        help='Sync transmission period in seconds')
    search.add_argument('--hops-rru1',
                        type=int,
                        help='Number of hops for RRU 1')
    search.add_argument('--hops-rru2',
                        type=int,
                        help='Number of hops for RRU 2')
    search.add_argument('--n-rru-ptp',
                        help='Number of RRUs actively operating as PTP slaves')
    search.add_argument('--delay-cal',
                        choices=['True', 'False'],
                        help='Whether dataset runs delay asymmetry calibration \
                        procedure')
    search.add_argument('--start-time',
                        type=str,
                        help='Dataset start time')
    search.add_argument('--pipeline-bbu',
                        type=int,
                        help='CI pipeline of the BBU bitstream')
    search.add_argument('--pipeline-rru',
                        type=int,
                        help='CI pipeline of the RRU bitstream')
    search.add_argument('--fh-traffic',
                        choices=['True', 'False'],
                        help='Whether or not FH traffic is active')
    search.add_argument('--fh-type',
                        choices=["inline", "cross"],
                        help='Fronthaul traffic type')
    search.add_argument('--fh-fs',
                        type=int,
                        choices=[7680000, 30720000],
                        help='LTE sample rate')
    search.add_argument('--fh-iq-size-dl',
                        type=int,
                        choices=list(range(4,34,2)),
                        help='IQ samples size in downlink')
    search.add_argument('--fh-iq-size-ul',
                        type=int,
                        choices=list(range(4,34,2)),
                        help='IQ samples size in uplink')
    search.add_argument('--fh-bitrate-dl',
                        help='Fronthaul bitrate in downlink')
    search.add_argument('--fh-bitrate-ul',
                        help='Fronthaul bitrate in uplink')
    search.add_argument('--fh-n-spf-dl',
                        type=int,
                        help='Number of IQ samples per frame in downlink')
    search.add_argument('--fh-n-spf-ul',
                        type=int,
                        help='Number of IQ samples per frame in uplink')
    search.add_argument('--fh-n-rru-dl',
                        type=int,
                        help='Number of RRUs that the BBU is configured to deliver \
                        data to in DL')
    search.add_argument('--fh-n-rru-ul',
                        type=int,
                        help='Number of RRUs delivering UL data, i.e. that are \
                        actually active in the testbed')
    search.add_argument('--fh-vlan-pcp',
                        type=int,
                        help='802.1Q priority code point (PCP) of FH frames')
    args = parser.parse_args()

    if (args.command is None):
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(stream=sys.stderr, level='INFO')


    ds_manager = ptp.datasets.Datasets()
    if (args.command == 'download'):
        ds_manager.download(args.file)

    if (args.command == 'search'):
        parameters = vars(args)
        parameters.pop('command')
        ds_found = ds_manager.search(parameters)

        if (ds_found):
            print("Found datasets:")
            df = pd.DataFrame(ds_found)
            df = df.sort_values(by=['start-time'])
            df = df.reset_index(drop=True)
            print(tabulate(df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
    main()
