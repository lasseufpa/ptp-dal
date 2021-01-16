#!/usr/bin/env python3
import argparse, sys, logging
import ptp.datasets


def main():
    parser    = argparse.ArgumentParser(description="Manage datasets")
    subparser = parser.add_subparsers()

    # Download datasets
    download = subparser.add_parser('download',
                                    help="Download datasets")
    download.add_argument('--file', '-f',
                          help='Dataset file path')

    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level")
    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    ds_manager = ptp.datasets.Datasets()
    ds_manager.download(args.file)


if __name__ == "__main__":
    main()
