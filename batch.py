#!/usr/bin/env python

"""Run batch of acquisitions
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess, json, sys, logging
from multiprocessing.pool import ThreadPool


def _append_key_val(cmd, key, val):
    """Add key,val pair from dictionary to a given list of arguments"""
    assert(len(key) > 0), "Empty key"

    if (isinstance(val, tuple)):
        # Special case used for counted argument (with action='count')
        keyarg = "-" + val[0] * val[1]
    elif (len(key) == 1):
        keyarg = "-" + key
    else:
        keyarg = "--" + key

    # The arguments come from a dictionary, so they should never be repeated
    assert(keyarg not in cmd)
    cmd.append(keyarg)

    if (val is None or isinstance(val, tuple)):
        return
    elif (isinstance(val, list)):
        for e in val:
            cmd.append(str(e))
    elif (val != ""):
        cmd.append(str(val))


def _run(action, arg_dict, job_id, dry_run=False):
    """Run the acquisition"""
    assert(isinstance(arg_dict, dict))
    logging.info("Running {} {}".format(action + ".py", job_id))

    # Convert argument dictionary to a list of arguments
    args = []
    for k,v in arg_dict.items():
        _append_key_val(args, k, v)

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
                   choices=["analyze"],
                   default="analyze",
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
    global_dict = cfg['global']

    # Add log level
    if (args.verbose > 1):
        # Special case: use tuple to indicate argument with action=count.
        # TODO: find a better solution
        global_dict['verbose'] = ("v", (args.verbose - 1))

    # Define the command-line arguments of each job
    starmap_args = list()
    for i, specific_dict in enumerate(cfg['batch']):
        # Join process-specific command-line arguments with global
        # arguments. The specific arguments can override the global args.
        arg_dict = {**global_dict, **specific_dict}
        starmap_args.append((args.action, arg_dict, i, args.dry_run))

    # Run the jobs in parallel or sequentially
    if args.jobs > 1:
        with ThreadPool(processes=args.jobs) as pool:
            pool.starmap(_run, starmap_args)
    else:
        for args in starmap_args:
            _run(*args)


if __name__ == "__main__":
    main()
