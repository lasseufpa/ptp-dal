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
    ENABLE_DELAY_CAL  = "15"
    DISABLE_DELAY_CAL = "16"


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
        assert(self.role in ["bbu", "rru"])

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
    def __init__(self, metadata, config, rru, rru2, bbu):
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
        self.metadata    = metadata
        self.config      = config

        # Devices that are active in the acquisition:
        self.all_dev = [self.bbu, self.rru]
        if self.rru2.active:
            self.all_dev.append(self.rru2)

        self._fh_ready_barrier = threading.Barrier(len(self.all_dev))
        self._sync_ready_barrier = threading.Barrier(len(self.all_dev))
        self._fh_traffic_ready_barrier = threading.Barrier(len(self.all_dev))
        self._cfg_complete = threading.Barrier(len(self.all_dev))
        self.prog_failed = False

    def _wait_sync_stage(self, device, target_sync_stage=3, cmd_timeout=10):
        """Wait until an RRU device achieves a target sync stage

        Args:
            device            : Device to check the sync state
            target_sync_stage : Syncronization state where capture should start
            cmd_timeout       : How long until a GET_SYNC_STAGE command timeout

        """
        logger.info("Waiting sync stage {} on {}".format(target_sync_stage,
                                                         device.name))

        sync_wait = True
        while sync_wait:
            device.send_cmd(Cmd.GET_SYNC_STAGE)

            s_time = time.time()
            while (True):
                line = device.read_line()

                if "Sync stage:" in line:
                    break

                c_time = time.time()
                if ((c_time - s_time) > cmd_timeout):
                    logger.warning("GET_SYNC_STAGE timeout")
                    break

            try:
                sync_stage = int(line.split(" ")[2])
            except ValueError:
                pass
            else:
                logger.debug(f"{device.name} sync stage: {sync_stage}")
                if sync_stage == target_sync_stage:
                    logger.info("Target sync stage acquired")
                    sync_wait = False

            time.sleep(2)

    def _wait_ad9361(self, device):
        """Waits until AD9361 is successfully initialized

        Args:
            device : Device to check AD9361 initialization

        """
        logger.info(f"Waiting AD9361 initialization on {device.name}")

        wait = True
        while wait:
            line = device.read_line()

            if ("AD9361" in line) and ("successfully" in line):
                logger.info(f"{device.name} AD9361 initialized")
                wait = False

    def _wait_fh_traffic(self, device, occ_interval=[4000, 4200], timeout=10):
        """Waits until the device succesfully initializes the FH traffic

        This is based on the device's occupancy that is read via UART. If the
        occupancy reaches the pre-specified interval, it is inferred that the FH
        traffic was initialized correctly.

        Args:
            device       : Device to check FH traffic
            occ_interval : Occupancy interval to check
            timeout      : Give up after this interval

        """
        logger.info(f"Checking FH Rx traffic of {device.name}")

        s_time = time.time()
        while True:
            line = device.read_line()
            if "Occupancy" in line:
                line_val = line[line.find("Occupancy:"):]
                line_val = line.split()
                if len(line_val) >= 4:
                    try:
                        occ_val = int(line_val[3])
                    except ValueError:
                        pass
                    if (occ_val >= occ_interval[0]) and (occ_val <= occ_interval[1]):
                        logger.info(f"FH Rx traffic of {device.name} is active")
                        break

            # Throw error if occupancy doesn't become right after a timeout
            c_time = time.time()
            if ((c_time - s_time) > timeout):
                raise RuntimeError(f"{device.name} occupancy did not "
                                   "reach expected range")

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

        if self.metadata['oscillator'] == "ocxo" and device.role == "rru":
            bitstream = "rru-sma-mgt"
        else:
            bitstream = device.role

        # Subprocess command
        command = ["python3", "prog.py", "-p", pipeline]

        if device.role is not 'bbu':
            command.append("-r")
            command.append(rru_n)

        if self.config['elf_only']:
            command.append("-e")

        command.append(bitstream)

        logger.debug("Running: " + " ".join(command))

        process = subprocess.Popen(
            command, cwd=prog_dir, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        process.communicate()
        process.wait()

        assert (process.returncode == 0), \
            "Error while programming the {} - return code {}".format(
                device.name, process.returncode
            )
        logger.info(f"Finished programing the {device.name}")

    def _config_device(self, device):
        """Configure the RoE device by sending RoE commands over UART

        Args:
            device      : Device serial connection

        """
        # Disable RFS
        logger.info(f"Disabling RFS : {device.name}")
        device.send_cmd(Cmd.DISABLE_RFS)
        time.sleep(1)

        if device is not self.bbu:
            # Switch clk8k RTC source from PTP to PPS
            logger.info(
                f"Switching clk8k RTC source from PTP to PPS : {device.name}"
            )
            device.send_cmd(Cmd.SWITCH_CLK8K_SRC)

            # Wait AD9361's initialization
            self._wait_ad9361(device)

        # Wait other threads
        self._fh_ready_barrier.wait()

        # Check Sync stage
        if device is not self.bbu:
            self._wait_sync_stage(device)

        # Wait until all RRUs reach the target Sync stage
        self._sync_ready_barrier.wait()

        # For delay asymmetry calibration, it is the BBU that coordinates the
        # enabling of FH traffic, and it is only the BBU that needs to enter
        # delay calibration mode. The RRUs will follow via control message.
        if self.metadata["delay_cal"]:
            if (device.name == "bbu"):
                device.send_cmd(Cmd.ENABLE_DELAY_CAL)
        # Enable Fronthaul Traffic
        elif self.metadata["fh_traffic"]:
            if (device.name == "bbu"):
                time.sleep(1) # enable DL after UL
                logger.info(f"Enabling DL traffic from {device.name}")
            else:
                logger.info(f"Enabling UL traffic from {device.name}")
            # Enable FH Tx traffic
            device.send_cmd(Cmd.ENABLE_FH)
            # Check FH Rx traffic
            self._wait_fh_traffic(device)

        self._fh_traffic_ready_barrier.wait()

        # Enable timestamp acquisition
        if device is not self.bbu:
            logger.info(f"Enabling timestamp acquisition on {device.name}")
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
            time.sleep(5) # let the processors run their initialization

        # Device Configuration
        if self.config["roe_configure"]:
            self._config_device(device)

        # All devices should be able to reach the end of their programming and
        # configuration threads. If not, it means there was a failure.
        try:
            self._cfg_complete.wait(timeout=10)
        except threading.BrokenBarrierError:
            self.prog_failed = True

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

        # Initializing threads
        threads = list()
        for device in self.all_dev:
            # Let the BBU thread start earlier, since it takes longer to load
            # the BBU elf
            if (device.name != "bbu"):
                time.sleep(10)
            thread = threading.Thread(target=self._program_and_config_thread,
                                      args=(device,),
                                      daemon=True)
            thread.start()
            threads.append(thread)

        if (self.prog_failed):
            raise RuntimeError("RoE programming failed")

        # Wait completion on all threads
        for thread in threads:
            thread.join()

        # Clear all buffers before capture
        for device in self.all_dev:
            device.clear_buffer()


