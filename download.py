#!/usr/bin/env python3
import subprocess, os, argparse


def download_dataset(path, remote_repo):
    """Download dataset from lasse100 machine

    Args:
        path        : dataset file path
        remote_repo : url of remote repo to scp from

    Returns:
        Boolean indicating whether file was found and downloaded

    """
    cmd = ["scp", os.path.join(remote_repo, path), "data/"]

    try:
        print("> %s" %(" ".join(cmd)))
        out   = subprocess.check_output(cmd)
        found = True
    except subprocess.CalledProcessError as e:
        found = False
        pass

    if (found):
        print("Downloaded")
        print("Run:\n./reader_demo.py -vvvv -f data/%s" %(path))
    else:
        print("Couldn't find dataset in %s" %(remote_repo))

    return found


def main():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument('-r', '--remote',
                        default="200.239.93.44",
                        help="Remote host where dataset is stored")
    parser.add_argument('-u', '--user',
                        default="roe",
                        help="User used to log into the remote host over ssh")
    parser.add_argument('file',
                        help='Dataset file path.')
    args     = parser.parse_args()

    print("Try to download %s from %s" %(args.file, args.remote))

    prefix      = args.user + "@" + args.remote
    global_repo = prefix + ":/opt/ptp_datasets/"
    user_repo   = prefix + ":/home/" + args.user + "/src/ptp_simulator/data/"
    repos       = [global_repo, user_repo]

    if (not os.path.exists(os.path.join("data", args.file))):
        for repo in repos:
            downloaded = download_dataset(args.file, repo)
            if (downloaded):
                return True
    else:
        print("Dataset already exists")


if __name__ == "__main__":
    main()
