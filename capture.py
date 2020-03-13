#!/usr/bin/env python

"""Acquisition of testbed data via serial

"""
import argparse, configparser, logging, sys, os, json, subprocess
from datetime import datetime
from pprint import pprint
import ptp.serial


def calc_rate(n_spf, l_iq, fs, n_rru_dl, n_rru_ul):
    """Compute theoretical DL/UL bitrates

    Args:
        n_spf    : Number of IQ samples per Ethernet frame
        l_iq     : IQ sample size (in bits)
        n_rru_dl : Number of RRUs configured at the BBU (determines DL rate)
        n_rru_ul : Number of RRUs that are actually active (determines UL rate)

    Returns:
        Tuple with DL and UL bitrates

    """
    # Constants
    eth_hdr_len    = 14*8
    fh_hdr_len     = 12*8
    n_axc_per_rru  = 2

    # Fronthaul frame length in bits
    fh_payload_len = n_spf * l_iq
    fh_frame_len   = eth_hdr_len + fh_hdr_len + fh_payload_len
    # Fronthaul frame transmission period
    i_bg           = n_spf / (n_axc_per_rru * fs);
    # Bitrate per RRU
    rate_per_rru   = fh_frame_len / i_bg
    # Total DL/UL bitrates
    bitrate_dl     = rate_per_rru * n_rru_dl
    bitrate_ul     = rate_per_rru * n_rru_ul

    return bitrate_dl, bitrate_ul


def get_pipeline(roe_path, pipeline):
    """Download the target CI pipeline

    If the pipeline is given as string "latest", download the latest successful
    pipeline build on roe_vivado's master branch.

    Args:
        roe_path        : Gitlab API directory path
        target_device   : Device to filter the pipeline's jobs

    Returns:
        pipeline : Pipeline number as string

    """
    roe_automation_path = os.path.join(roe_path, "automation")
    assert os.path.exists(roe_automation_path), \
        "Couldn't find path {}".format(roe_automation_path)

    command = ['python3', 'gitlab-download.py']

    if pipeline != "latest":
        command.extend(['-p', pipeline])

    output = subprocess.check_output(command, cwd=roe_automation_path)
    lines = output.decode().splitlines()
    for line in lines:
        if (("Latest pipeline" in line) and (len(line.split()) > 5)):
            pipeline = line.split()[5]
    assert(int(pipeline)) # assert it can be converted to int
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Capture timestamps from FPGA")
    parser.add_argument('--rru',
                        default="rru_uart",
                        choices=["rru_uart", "rru2_uart"],
                        help='Target char device for UART communication with \
                        an RRU FPGA (default: rru_uart).')
    parser.add_argument('--rru2',
                        default="rru2_uart",
                        choices=["rru_uart", "rru2_uart"],
                        help='Target char device for UART communication with \
                        an RRU2 FPGA (default: rru2_uart).')
    parser.add_argument('--bbu',
                        default="bbu_uart",
                        help='Target char device for UART communication \
                        with the BBU FPGA (default: bbu_uart).')
    parser.add_argument('--sensor',
                        default="roe_sensor",
                        help='Target char device for UART communication \
                        with the (Arduino) device that hosts the \
                        temperature sensor (default: roe_sensor).')
    parser.add_argument('-N', '--num-iter',
                        default=0,
                        type=int,
                        help='Restrict number of iterations. If set to 0, the \
                        acquisition will run indefinitely (default: 0).')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        default=1,
                        help="Verbosity (logging) level.")
    parser.add_argument('--oscillator',
                        default="xo",
                        choices=["ocxo", "xo"],
                        help='Define the oscillator type')
    parser.add_argument('--sync-rate',
                        default=4,
                        type=float,
                        help='Sync transmission period in seconds')
    parser.add_argument('--hops',
                        type=int,
                        default=4,
                        help='Number of hops')
    parser.add_argument('--n-rru-ptp',
                        type=int,
                        default=2,
                        help='Number of RRUs actively operating as PTP slaves \
                        in the testbed (not necessarily delivering UL FH data)')

    fh_traffic_group = parser.add_argument_group('background traffic')
    fh_traffic_group.add_argument('--fh-traffic',
                                  default=False,
                                  action='store_true',
                                  help='Whether or not FH traffic is active')
    fh_traffic_group.add_argument('--type',
                                  choices=["inline","cross"],
                                  default="inline",
                                  help='Fronthaul traffic type')
    fh_traffic_group.add_argument('--fs',
                                  type=float,
                                  choices=[7680000, 30720000],
                                  default=7680000,
                                  help='LTE sample rate')
    fh_traffic_group.add_argument('--iq-size',
                                  type=int,
                                  choices=list(range(4,34,2)),
                                  default=24,
                                  help='IQ samples size')
    fh_traffic_group.add_argument('--n-spf',
                                  type=int,
                                  default=64,
                                  help='Number of IQ samples per frame')
    fh_traffic_group.add_argument('--n-rru-dl',
                                  type=int,
                                  default=2,
                                  help='Number of RRUs that the BBU is \
                                  configured to deliver data to in DL')
    fh_traffic_group.add_argument('--n-rru-ul',
                                  type=int,
                                  default=2,
                                  help=("Number of RRUs delivering UL data, "
                                        "i.e. that are actually active in the "
                                        "testbed"))

    roe_prog_config_group = parser.add_argument_group(('RoE programing and '
                                                       'configuration'))
    roe_prog_config_group.add_argument('-p', '--roe-prog',
                                       default=False,
                                       action='store_true',
                                       help=("Set in order to program the FPGAs"
                                             " before the acquisition"))
    roe_prog_config_group.add_argument('-e', '--roe-elf-only',
                                       default=False,
                                       action='store_true',
                                       help=("Set in order to load only the "
                                             "into the FPGAs, and not the "
                                             "bitstream"))
    roe_prog_config_group.add_argument('--roe-rebuild',
                                       default=False,
                                       action='store_true',
                                       help=("Set in order to rebuild the elf "
                                             "before programming"))
    roe_prog_config_group.add_argument('-c', '--roe-configure',
                                       default=False,
                                       action='store_true',
                                       help=("Set to enable automatic runtime "
                                             "configuration of RoE devices"))
    roe_prog_config_group.add_argument('--bbu-pipeline',
                                       default="latest",
                                       type=str,
                                       help='CI pipeline of the BBU bitstream')
    roe_prog_config_group.add_argument('--rru-pipeline',
                                       default="latest",
                                       type=str,
                                       help='CI pipeline of the RRU bitstream')
    roe_prog_config_group.add_argument('--roe-path',
                                       default="../roe_vivado/",
                                       type=str,
                                       help='Path to `roe_vivado` directory')
    args = parser.parse_args()

    logging_level = 70 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    # If background traffic is active, create a dictionary with all
    # information to save as metadata
    if (args.fh_traffic) :
        # Compute theoretical DL/UL bitrates
        bitrate_dl, bitrate_ul = calc_rate(args.n_spf, args.iq_size, args.fs,
                                           args.n_rru_dl, args.n_rru_ul)

        fh_traffic = {
            "type" : args.type,
            "fs"   : args.fs, # in Hz
            "bitrate" : { # in bps
                "dl" : bitrate_dl,
                "ul" : bitrate_ul,
            },
            "iq_size" : args.iq_size,
            "n_spf" : args.n_spf,
            "n_rru_ul" : args.n_rru_ul,
            "n_rru_dl" : args.n_rru_dl
        }
    else:
        fh_traffic = None

    # Parse pipeline
    args.bbu_pipeline = get_pipeline(args.roe_path, args.bbu_pipeline)
    args.rru_pipeline = get_pipeline(args.roe_path, args.rru_pipeline)

    # Dictionary for RoE programing and configuration metadata
    if (args.roe_prog) or (args.roe_configure):
        assert os.path.exists(args.roe_path), \
            "Couldn't find roe_vivado directory"

        pipelines = {
            "bbu" : args.bbu_pipeline,
            "rru" : args.rru_pipeline,
        }

        roe_prog_config = {
            "roe_prog"        : args.roe_prog,
            "roe_configure"   : args.roe_configure,
            "pipeline"        : pipelines,
            "roe_vivado_path" : args.roe_path,
            "elf_only"        : args.roe_elf_only,
            "rebuild_elf"     : args.roe_rebuild
        }

    else:
        roe_prog_config = None
        pipelines       = None

    # Dictionary containing the metadata
    metadata = {
        "oscillator": args.oscillator,
        "sync_period": 1.0/args.sync_rate,
        "fh_traffic" : fh_traffic,
        "hops" : args.hops,
        "n_rru_ptp" : args.n_rru_ptp,
        "pipelines" : pipelines,
        "start_time" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print("Metadata:")
    pprint(metadata)
    raw_resp = input("Proceed? [Y/n] ") or "Y"
    response = raw_resp.lower()

    if (response.lower() == "y"):
        serial = ptp.serial.Serial(args.rru, args.rru2, args.bbu, args.sensor,
                                   args.num_iter, metadata, roe_prog_config)
        serial.run()

if __name__ == "__main__":
    main()
