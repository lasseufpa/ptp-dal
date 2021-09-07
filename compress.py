#!/usr/bin/env python
import logging
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import ptp.compression


def parser():
    parser = ArgumentParser(description="Compress JSON dataset",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', help='JSON or Pickle dataset file.')
    parser.add_argument('-f',
                        '--format',
                        default="xz",
                        choices=["json", "pickle", "gz", "pbz2", "xz"],
                        help='Output file format. Determines also the binary \
                        compression application that is used on top of the \
                        more efficient data representation that is obtained \
                        by our PTP compression module.')
    parser.add_argument('--verbose',
                        '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level")
    args = parser.parse_args()
    return args


def main():
    args = parser()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    codec = ptp.compression.Codec(filename=args.file)
    codec.compress()
    codec.dump("json")
    codec.dump("pickle")
    codec.dump("gz")
    codec.dump("pbz2")
    codec.dump("xz")


if __name__ == "__main__":
    main()
