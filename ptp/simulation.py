"""PTP simulation

Conventions:
- Simulation time is given in seconds
- RTC time is kept in seconds/nanoseconds
- Message periods are in seconds
- units are explicit within variable names where possible

"""
import logging
import heapq
import random
import time
from pathlib import Path
from pprint import pprint

import ptp.compression
from ptp.rtc import Rtc
from ptp.messages import PtpEvt
from ptp.mechanisms import DelayReqResp

logger = logging.getLogger(__name__)


class SimTime():
    def __init__(self, t_step):
        """Simulation Time

        Keeps track of simulation time in seconds

        Args:
            t_step : Simulation time step in seconds
        """
        self.time = 0
        self.t_step = t_step

    def get_time(self):
        """Return the simulation time"""
        return self.time

    def advance(self, next_time):
        """Advance simulation time to a specified instant"""
        self.time = next_time
        logger.debug("Advance simulation time to: %f ns" % (self.time * 1e9))

    def step(self):
        """Advance simulation time by the simulation step"""
        self.time += self.t_step


class Simulation():
    """PTP Simulation class

    Args:
        n_iter          : Number of iterations
        sim_t_step      : Simulation time step in seconds
        sync_period     : Sync transmission period in seconds
        rtc_clk_freq    : RTC clock frequency in Hz
        rtc_resolution  : RTC representation resolution in ns
        freq_tolerance  : Slave RTC frequency tolerance in ppb
        freq_rw         : Normalized variance of the frequency offset
                            random-walk presented by the slave RTC.
        phase_rw        : Normalized variance of the phase offset
                            random-walk presented by the slave RTC.
        pdv_distr       : PTP message PDV distribution (Gamma or Gaussian)
        gamma_shape     : Shape parameter of the Gamma distribution
        gamma_scale     : Scale parameter of the Gamma distribution
        ts_quantization : Enables quantization of the time scale

    """
    def __init__(self,
                 n_iter=100,
                 sim_t_step=1e-9,
                 sync_period=1.0 / 16,
                 rtc_clk_freq=125e6,
                 rtc_resolution=0,
                 freq_tolerance=60,
                 freq_rw=1e-18,
                 phase_rw=1e-12,
                 pdv_distr="Gamma",
                 gamma_shape=None,
                 gamma_scale=None,
                 ts_quantization=True):
        self.n_iter = n_iter
        self.sync_period = sync_period
        self.rtc_clk_freq = rtc_clk_freq
        self.rtc_resolution = rtc_resolution
        self.pdv_distr = pdv_distr
        self.slave_freq_tolerance = freq_tolerance
        self.slave_freq_rw = freq_rw
        self.slave_phase_rw = phase_rw
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.ts_quantization = ts_quantization

        # Simulation time
        self.sim_timer = SimTime(sim_t_step)

        # Progress
        self.last_progress_print = 0

        # Simulation data and metadata
        self.data = list()
        self.metadata = {}

    def check_progress(self, i_iter):
        """Check/print simulation progress"""

        progress = i_iter / self.n_iter

        if (progress > self.last_progress_print + 0.1):
            print("Progress: %6.2f %%" % (progress * 100))
            self.last_progress_print = progress

    def save(self):
        """Save simulation data and metadata on compressed file"""

        path = "data/"
        filename = path + "sim-" + time.strftime("%Y%m%d-%H%M%S")

        # Collect metadata
        self.metadata = {
            'n_iter': self.n_iter,
            'sync_period': self.sync_period,
            'rtc_clk_freq': self.rtc_clk_freq,
            'rtc_resolution': self.rtc_resolution,
            'pdv_distr': self.pdv_distr,
            'slave_freq_tolerance': self.slave_freq_tolerance,
            'slave_freq_rw': self.slave_freq_rw,
            'slave_phase_rw': self.slave_phase_rw,
            'gamma_shape': self.gamma_shape,
            'gamma_scale': self.gamma_scale,
            'ts_quantization': self.ts_quantization
        }

        # Dataset
        ds = {'metadata': self.metadata, 'data': self.data}

        # Compress and save to file
        codec = ptp.compression.Codec(ds, filename)
        codec.compress()
        codec.dump(ext="xz")

    def load(self, filename):
        """Load simulation data and metadata from compressed file

        Args:
            filename : Path to the dataset

        """
        assert (Path(filename).exists()), "Load file does not exist"
        codec = ptp.compression.Codec(filename=filename)
        ds = codec.decompress()
        self.data = ds['data']
        self.metadata = ds['metadata']
        logger.info("Imported data from %s" % (filename))

    def dump(self):
        """Dump simulation metadata and data into stdout

        Simulation data is printed according to the logging level.

        """
        print("Simulation configurations:")
        pprint(self.metadata)

        logger.info("Simulation data:")
        DelayReqResp.log_header(level=logging.INFO)
        for x in self.data:
            DelayReqResp.log(x, level=logging.INFO)

    def run(self):
        """Main loop

        Simulates PTP delay request-response message exchanges with
        corresponding time offset and delay estimations.

        """

        # Register the PTP message objects
        sync = PtpEvt("Sync",
                      self.sync_period,
                      pdv_distr=self.pdv_distr,
                      gamma_shape=self.gamma_shape,
                      gamma_scale=self.gamma_scale)
        dreq = PtpEvt("Delay_Req",
                      pdv_distr=self.pdv_distr,
                      gamma_shape=self.gamma_shape,
                      gamma_scale=self.gamma_scale)

        # RTCs
        #
        # NOTE: impairment parameters for the Master are omitted because the
        # default is to assume perfect conditions (no phase/freq. noise and no
        # freq. error).
        master_rtc = Rtc(self.rtc_clk_freq,
                         self.rtc_resolution,
                         label="Master",
                         ts_quantization=self.ts_quantization)
        slave_rtc = Rtc(self.rtc_clk_freq,
                        self.rtc_resolution,
                        tol_ppb=self.slave_freq_tolerance,
                        norm_var_freq_rw=self.slave_freq_rw,
                        norm_var_time_rw=self.slave_phase_rw,
                        label="Slave",
                        ts_quantization=self.ts_quantization)

        # Main loop
        evts = list()
        stop = False
        i_iter = 0
        dreqresps = list()

        # Start with a sync transmission
        sync.next_tx = 0

        DelayReqResp.log_header()

        while (not stop):
            sim_time = self.sim_timer.get_time()

            # Update the RTCs
            master_rtc.update(sim_time, evts)
            slave_rtc.update(sim_time, evts)

            # Try processing all events
            sync_transmitted = sync.tx(sim_time, master_rtc.get_time(), evts)
            dreq_transmitted = dreq.tx(sim_time, slave_rtc.get_time(), evts)
            sync_received = sync.rx(sim_time, slave_rtc.get_time(),
                                    master_rtc.get_time())
            dreq_received = dreq.rx(sim_time, master_rtc.get_time(),
                                    slave_rtc.get_time())

            # Post-processing for each message
            if (sync_transmitted):
                # Save Sync departure timestamp
                dreqresp = DelayReqResp(sync.seq_num, sync.tx_tstamp)
                dreqresps.insert(sync.seq_num, dreqresp)

            if (sync_received):
                # Save Sync arrival timestamp
                dreqresp = dreqresps[sync.seq_num]
                dreqresp.set_t2(sync.seq_num, sync.rx_tstamp)
                # Save the true one-way delay
                dreqresp.set_forward_delay(sync.seq_num, sync.one_way_delay)
                # Schedule the Delay_Req transmission after a random delay
                rndm_t2_to_t3 = random.gauss(70 * 1e-6, 2e-6)
                # NOTE: our testbed measures around 70 microseconds
                dreq.sched_tx(sim_time + rndm_t2_to_t3, evts)

            if (dreq_transmitted):
                # Save Delay_Req departure timestamp
                dreqresp = dreqresps[dreq.seq_num]
                dreqresp.set_t3(dreq.seq_num, dreq.tx_tstamp)

            if (dreq_received):
                # Save Delay_Req arrival timestamp
                dreqresp = dreqresps[dreq.seq_num]
                dreqresp.set_t4(dreq.seq_num, dreq.rx_tstamp)
                # Save the true one-way delay
                dreqresp.set_backward_delay(dreq.seq_num, dreq.one_way_delay)
                # Define true time offset and asymmetry
                dreqresp.set_true_toffset(master_rtc.get_time(),
                                          slave_rtc.get_time())

                # Process all four timestamps
                results = dreqresp.process()
                dreqresp.log(results)

                # Include RTC state
                results["rtc_y"] = slave_rtc.get_freq_offset()

                # Append to all-time simulation data
                self.data.append(results)

                # Message exchange count
                i_iter += 1

            # Update simulation time
            if (len(evts) > 0):
                next_evt = heapq.heappop(evts)
                self.sim_timer.advance(next_evt)
            else:
                self.sim_timer.step()

            # Stop criterion
            if (i_iter >= self.n_iter):
                print("Progress: %6.2f %%" % (100))
                stop = True

            self.check_progress(i_iter)
