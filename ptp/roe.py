import subprocess
import time
import json
import logging
import serial
import os
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class Cmd(Enum):
    """Commands accepted by the RoE device"""
    GET_SYNC_STAGE    = "1"
    ENABLE_TS_CAPTURE = "2"
    DISABLE_RFS       = "6"
    ENABLE_FH         = "8"
    SWITCH_CLK8K_SRC  = "14"


class RoE_device():
    def __init__(self, serial, name, active=True):
        """RoE device management

        Args:
            serial : Serial connection
            name   : Device's name
            active : Whether the device is active in the FH

        """
        self._serial = serial
        self.name    = name
        self.active  = active
        self.role    = name[:3]
        assert(self.type in ["bbu", "rru"])

    def read_line(self):
        """Readline and clean whitespaces"""

        line = self._serial.readline().strip().decode("utf-8", "ignore")
        return " ".join(line.split())

    def send_cmd(self, cmd):
        """Send command to device"""
        self._serial.write((cmd.value + "\n").encode())

    def clear_buffer(self):
        """Clear the device buffer"""
        self._serial.reset_input_buffer()


class RoE:
    def __init__(self, metadata, config, rru, rru2, bbu, sensor):
        """RoE manager class.

        This class is responsible for configuration and
        programming of the RoE devices.

        Args:
            metadata    : Acquisition metadata
            config      : Programing and configuration info
            rru         : RRU serial connection
            bbu         : BBU serial connection
            config_path : Path to the json configuration file

        """
        self.bbu         = RoE_device(bbu, 'bbu')
        self.rru         = RoE_device(rru, 'rru')
        self.rru2        = RoE_device(rru2, 'rru2',
                                      active=(metadata["n_rru_ptp"] == 2))
        self.sensor      = RoE_device(sensor,'sensor') \
                           if sensor is not None else None
        self.metadata    = metadata
        self.config      = config

        # Devices that are active in the acquisition:
        self.all_dev = [self.bbu, self.rru]
        if self.rru2.active:
            self.all_dev.append(self.rru2)

    def _wait_sync_stage(self, device, target_sync_stage=3):
        """Wait until an RRU device achieves a target sync stage

        Args:
            device            : Device to check the sync state
            target_sync_stage : Syncronization state where capture should start

        """
        logger.info(f"Checking sync stage of {device.name}")

        sync_wait       = True
        self.last_read  = time.time()

        while sync_wait:
            device.send_cmd(Cmd.GET_SYNC_STAGE)
            line = device.read_line()

            if "Sync" in line:
                try:
                    sync_stage = int(line.split(" ")[2])
                except ValueError:
                    pass
                else:
                    logging.info(f"{device.name} sync stage {sync_stage}")
                    if sync_stage == target_sync_stage:
                        logging.info("Target sync stage acquired")
                        sync_wait = False

            time.sleep(2)

    def _wait_ad9361(self, device):
        """Waits until AD9361 is successfully initialized

        Args:
            device : Device to check AD9361 initialization

        """
        logger.info(f"Waiting AD9361 initialization in {device.name}")

        wait = True
        while wait:
            line = device.read_line()

            if ("AD9361" in line) and ("successfully" in line):
                logging.info(f"{device.name} AD9361 initialized")
                wait = False

    def _wait_fh_traffic(self, device, occ_inter=[3796, 4396]):
        """Waits until the device succesfully initializes the FH traffic

        This is based on the device's occupancy that is read via UART. If the
        occupancy reaches the pre-specified interval, it is inferred that the FH
        traffic was initialized correctly.

        Args:
            device     : Device to check FH traffic
            occ_inter  : Occupancy interval to check

        """
        device.reset_input_buffer()
        logger.info(f"Checking FH traffic of {device.name}")

        wait = True
        time.sleep(0.5)
        while wait:
            line = device.read_line()
            if "Occupancy" in line:
                line_val = line[line.find("Occupancy:"):]
                line_val = line.split()
                if len(line_val) >= 4:
                    try:
                        occ_val = int(line_val[3])
                    except ValueError:
                        pass
                    logger.info(f"Occupancy in {device.name} {occ_val}")
                    if (occ_val >= occ_inter[0]) and (occ_val <= occ_inter[1]):
                        logger.info(f"{device.name} initialized FH traffic")
                        wait = False
                        break

    def _run_prog(self, device, pipeline):
        """Program the FPGA with the bitstream of a given pipeline

        Use the `prog.py` script that is located within the
        `roe_vivado/automation/` diretory.

        Args:
            device   : RoE device object to program
            pipeline : Gitlab CI/CD pipeline

        """
        rru_n       = '2' if device.name is 'rru2' else '1'
        prog_dir    = os.path.join(self.config["roe_vivado_path"], "automation")

        logger.info(f"Programing {device.name}")

        # Subprocess command
        command = ["python3", "prog.py", device.role, "-p", pipeline]

        if device.role is not 'bbu':
            command.append("-r")
            command.append(rru_n)

        self.process = subprocess.Popen(
            command, cwd=prog_dir, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        self.process.communicate()
        self.process.wait()

        assert (self.process.returncode == 0), \
            f"Error while programming {device.name}"
        logger.info(f"Finished programing {device.name}")
        time.sleep(0.5)

    def _config_device(self, device):
        """Configure the RoE device by sending RoE commands over UART

        Args:
            device      : Device serial connection

        """
        # Disable RFS
        logger.info(f"Disabling RFS : {device.name}")
        device.send_cmd(Cmd.DISABLE_RFS)
        time.sleep(4)

        if device is not self.bbu:
            # Switch clk8k RTC source from PPS to PTP
            logger.info(
                f"Switching clk8k RTC source from PPS to PTP : {device.name}"
            )
            device.send_cmd(Cmd.SWITCH_CLK8K_SRC)

            # Wait AD9361's initialization
            self._wait_ad9361(device)
            time.sleep(5)

        # Enable Fronthaul Traffic
        if self.metadata["fh_traffic"]:
            logger.info(f"Enabling FH traffic : {device.name}")
            device.send_cmd(Cmd.ENABLE_FH)
            self._wait_fh_traffic(device)

        # Check Sync stage
        if device is not self.bbu:
            self._wait_sync_stage(device)
            logger.info(f"Enabling timestamp capture : {device.name}")
            device.send_cmd(Cmd.ENABLE_TS_CAPTURE)

    def _program_and_config_thread(self, device):
        """Execute the programing and configuration routines

        Each device will have a programing and configuration thread, so that
        they can be carried out in parallel.

        Args:
            device  :  Target device for the settings

        """
        # Device Programing
        if self.config["roe_prog"]:
            self._run_prog(device, self.config["pipeline"][device.role])
            time.sleep(5)

        # Device Configuration
        if self.config["roe_configure"]:
            self._config_device(device)

    def prog_and_configure(self):
        """Program and configure RoE Devices

        Launches a separate thread for each device and runs the pramming and
        configuration steps in parallel.

        """
        if self.config['roe_prog']:
            # Prog script Directory
            prog_dir = self.config["roe_vivado_path"] + "/automation"
            assert os.path.exists(
                os.path.join(prog_dir, "prog.py")
            ), "Couldn't find prog.py"
            logger.info("Found prog.py script")

        # Initializing threads
        threads = list()
        for device in self.all_dev:
            thread = threading.Thread(target=self._program_and_config_thread,
                                      args=(device,),
                                      daemon=True)
            thread.start()
            threads.append(thread)
            # Add a time gap between the programming threads to avoid
            # irregular behaviors on RoE devices occupancy
            time.sleep(35)

        # Wait completion on all threads
        for thread in threads:
            thread.join()

        # Clear all buffers before capture
        for device in self.all_dev:
            device.clear_buffer()
        if self.sensor:
            self.sensor.clear_buffer()


