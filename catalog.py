#!/usr/bin/env python
import argparse, logging, sys
import ptp.docs

def main():
    parser = argparse.ArgumentParser(description="Catalog dataset files")

    parser.add_argument('-f', '--file',
                        help='JSON file to add to markdown table \
                        (default: %(default)s)')

    parser.add_argument('-d', '--directory',
                        default='/opt/ptp_datasets/',
                        help='Dataset directory (default: %(default)s)')

    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity level (default: %(default)s)")

    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    docs = ptp.docs.Docs(cfg_path=args.directory)

    if (args.file):
        # If the file was passed, just add the file to the markdown table
        docs.add_value(args.file)
    else:
        # Generate table with all datasets
        docs.process()

if __name__ == '__main__':
    main()
