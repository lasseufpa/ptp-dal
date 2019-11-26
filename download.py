#!/usr/bin/env python3
import subprocess, os, argparse


def download_dataset(path):
    """Download dataset from lasse100 machine

    Args:
        path : dataset file path

    """
    print("Try to download %s from 200.239.93.44" %(path))
    remote_repo = "roe@200.239.93.44:/home/roe/src/ptp_simulator/data/"
    cmd         = ["scp", os.path.join(remote_repo, path), "data/"]
    out         = subprocess.check_output(cmd)
    print(out.decode())
    print("Done")


def main():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument('file',
                        help='Dataset file path.')
    args     = parser.parse_args()

    if (not os.path.exists(os.path.join("data", args.file))):
        download_dataset(args.file)
    else:
        print("Dataset already exists")


if __name__ == "__main__":
    main()
