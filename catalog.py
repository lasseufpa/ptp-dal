#!/usr/bin/env python
import argparse, logging, sys
import ptp.docs

def main():
    parser = argparse.ArgumentParser(description="Catalog dataset files")

    parser.add_argument('-f', '--file',
                        default=None,
                        help='Dataset JSON file to be cataloged \
                        (default: %(default)s)')

    parser.add_argument('-d', '--directory',
                        default='/opt/ptp_datasets/',
                        help='Dataset directory (default: %(default)s)')

    parser.add_argument('-r', '--reset',
                        default=False,
                        action='store_true',
                        help="Reset catalog (default: %(default)s)")

    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity level (default: %(default)s)")

    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    docs = ptp.docs.Docs(cfg_path=args.directory, reset=args.reset)

    if (args.file):
        # If a specific dataset was given, just add it to the dataset
        docs.add_dataset(args.file)
    else:
        # Generate table with all datasets
        docs.process()

if __name__ == '__main__':
    main()
