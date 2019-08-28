import argparse, logging, sys
import ptp.docs

def main():
    parser = argparse.ArgumentParser(description="Generate documentation")
    parser.add_argument('-f', '--file',
                        help='JSON file to add in markdown table')
    parser.add_argument('--path',
                        default='/opt/ptp_datasets/',
                        help='Path with datasets')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level")
    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    docs = ptp.docs.Docs(cfg_path=args.path)

    if (args.file):
        # If the file was passed, just add the file to the markdown table
        docs.add_value(args.file)
    else:
        # Generate table with all datasets
        docs.process()

if __name__ == '__main__':
    main()