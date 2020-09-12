#!/usr/bin/env python

"""Run batch of acquisitions
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess, json, sys, logging
from multiprocessing.pool import ThreadPool


def _append_key_val(cmd, key, val):
    """Add key,val pair from dictionary to a given list of arguments"""
    if (len(key) == 1):
        cmd.append("-" + key)
    else:
        cmd.append("--" + key)

    if (val is None):
        return
    elif (isinstance(val, list)):
        for e in val:
            cmd.append(str(e))
    elif (val != ""):
        cmd.append(str(val))


def _run(action, args, job_id, dry_run=False):
    """Run the acquisition"""
    logging.info("Running {} {}".format(action + ".py", job_id))

    script = action + ".py"
    cmd    = ["python3", script]
    cmd.extend(args)

    if (dry_run):
        print(" ".join(cmd))
        return

    logging.debug("> " + " ".join(cmd))

    res = subprocess.run(cmd)
    res.check_returncode()


def parser():
    p = ArgumentParser(description="Run a batch of actions",
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('filename',
                   help='JSON file describing the target action batch')
    p.add_argument('-a','--action',
                   choices=["capture", "analyze"],
                   default="capture",
                   help="Action to run in batch (capture or analyze).")
    p.add_argument('-j', '--jobs',
                   type=int,
                   default=1,
                   help="Number jobs to run concurrently.")
    p.add_argument('--verbose', '-v',
                   action='count',
                   default=1,
                   help="Verbosity (logging) level.")
    p.add_argument('--dry-run',
                   action='store_true',
                   default=False,
                   help="Print all commands but do not run them")
    return p.parse_args()


def main():
    args = parser()

    # It's not possible to run the "capture" action with parallel jobs
    assert(args.jobs == 1 or args.action=="analyze")

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    with open(args.filename, 'r') as fd:
        cfg = json.load(fd)

    # Validate JSON format
    #
    # The root of the dictionary should contain the 'global' and 'batch'
    # keys. The 'global' entry should be a dictionary holding the global
    # command-line arguments. The 'batch' entry should be a list of
    # dictionaries, each describing the specific parameters of the action.
    assert('global' in cfg)
    assert('batch' in cfg)
    assert(isinstance(cfg['global'], dict))
    assert(isinstance(cfg['batch'], list))
    assert(all([isinstance(x, dict) for x in cfg['batch']]))

    # Global command-line arguments
    log_level_arg = "-" + ((args.verbose - 1) * "v") if args.verbose > 1 else ""
    global_args   = [log_level_arg]
    for param in cfg['global']:
        _append_key_val(global_args, param, cfg['global'][param])

    # Define the command-line arguments of each job
    starmap_args = list()
    for i, action in enumerate(cfg['batch']):
        # Specific command-line arguments
        action_args = []
        action_args.extend(global_args)
        for param in action:
            _append_key_val(action_args, param, action[param])

        starmap_args.append((args.action, action_args, i, args.dry_run))

    # Run the jobs in parallel or sequentially
    if args.jobs > 1:
        with ThreadPool(processes=args.jobs) as pool:
            pool.starmap(_run, starmap_args)
    else:
        for args in starmap_args:
            _run(*args)


if __name__ == "__main__":
    main()
