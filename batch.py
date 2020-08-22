#!/usr/bin/env python

"""Run batch of acquisitions
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess, json, sys, logging


def _append_key_val(cmd, key, val):
    """Add key,val pair from dictionary to a given list of arguments"""
    if (len(key) == 1):
        cmd.append("-" + key)
    else:
        cmd.append("--" + key)

    if (val is not None and val != ""):
        cmd.append(str(val))


def _run(action, args, dry_run=False):
    """Run the acquisition"""
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
    log_level_arg = "-" + ((args.verbose - 1) * "v") if args.verbose > 1 else ""
    global_args   = [log_level_arg]
    for param in cfg['global']:
        _append_key_val(global_args, param, cfg['global'][param])

    for i, action in enumerate(cfg['batch']):
        # Specific command-line arguments
        action_args = []
        action_args.extend(global_args)
        for param in action:
            _append_key_val(action_args, param, action[param])

        logging.info("Running {} {}".format(args.action + ".py", i))

        # Run the action
        _run(args.action, action_args, args.dry_run)


if __name__ == "__main__":
    main()
