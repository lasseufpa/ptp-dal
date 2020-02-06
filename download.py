#!/usr/bin/env python3
import argparse, sys, logging
from ptp import download


def main():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument('file',
                        help='Dataset file path.')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level")
    args     = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)
    downloader = download.Download(args.file)
    downloader.run()


if __name__ == "__main__":
    main()
