"""PTP simulation runner

Conventions:
- Simulation time is given in seconds
- RTC time is kept in seconds/nanoseconds
- Message periods are in seconds
- units are explicit within variable names where possible

"""
import logging, heapq, random
from ptp.rtc import *
from ptp.messages import *
from ptp.mechanisms import *
from ptp.estimators import *


class SimTime():
    def __init__(self, t_step):
        """Simulation Time

        Keeps track of simulation time in seconds

        Args:
            t_step : Simulation time step in seconds
        """
        self.time   = 0
        self.t_step = t_step

    def get_time(self):
        """Return the simulation time"""
        return self.time

    def advance(self, next_time):
        """Advance simulation time to a specified instant"""
        self.time = next_time
        logger = logging.getLogger("SimTime")
        logger.debug("Advance simulation time to: %f ns" %(self.time*1e9))

    def step(self):
        """Advance simulation time by the simulation step"""
        self.time += self.t_step


class Runner():
    def __init__(self, n_iter = 100, sim_t_step = 1e-9, sync_period = 1.0/16,
                 rtc_clk_freq = 125e6, rtc_resolution = 0, rtc_tolerance = 60,
                 rtc_stability = 1.0, pdv_distr="Gamma", freq_est_per = 1e9):
        """PTP Runner class

        Args:
            n_iter         : Number of iterations
            sim_t_step     : Simulation time step in seconds
            sync_period    : Sync transmission period in seconds
            rtc_clk_freq   : RTC clock frequency in Hz
            rtc_resolution : RTC representation resolution in ns
            rtc_tolerance  : Slave RTC frequency tolerance in ppb
            rtc_stability  : Slave RTC freq. stability (0 for constant freq.)
            pdv_distr      : PTP message PDV distribution (Gamma or Gaussian)
            freq_est_per   : Raw freq. estimation period in ns

        """

        self.n_iter              = n_iter
        self.sync_period         = sync_period
        self.rtc_clk_freq        = rtc_clk_freq
        self.rtc_resolution      = rtc_resolution
        self.pdv_distr           = pdv_distr
        self.slave_rtc_tolerance = rtc_tolerance
        self.slave_rtc_stability = rtc_stability
        self.freq_est_per_ns     = freq_est_per

        # Simulation time
        self.sim_timer = SimTime(sim_t_step)

        # Progress
        self.last_progress_print = 0

        # Simulation data
        self.data = list()

    def check_progress(self, i_iter):
        """Check/print simulation progress"""

        progress = i_iter / self.n_iter

        if (progress > self.last_progress_print + 0.1):
            print("Runner progress: %f %%" %(progress * 100))
            self.last_progress_print = progress

    def run(self):
        """Main loop

        Simulates PTP delay request-response message exchanges with
        corresponding time offset and delay estimations.

        """

        # Register the PTP message objects
        sync = PtpEvt("Sync", self.sync_period, pdv_distr=self.pdv_distr)
        dreq = PtpEvt("Delay_Req", pdv_distr=self.pdv_distr)

        # RTC parameters
        master_ppb_tol       = 0 # master RTC is assumed perfect
        master_ppb_stability = 0 # master RTC is assumed perfect
        slave_ppb_tol        = self.slave_rtc_tolerance
        slave_ppb_stability  = self.slave_rtc_stability

        # RTCs
        master_rtc = Rtc(self.rtc_clk_freq, self.rtc_resolution, master_ppb_tol,
                         master_ppb_stability, "Master")
        slave_rtc  = Rtc(self.rtc_clk_freq, self.rtc_resolution, slave_ppb_tol,
                         slave_ppb_stability, "Slave")

        # Main loop
        evts       = list()
        stop       = False
        i_iter     = 0
        dreqresps  = list()

        # Estimators
        freq_estimator = FreqEstimator(self.freq_est_per_ns)

        # Start with a sync transmission
        sync.next_tx = 0

        DelayReqResp(0,0).log_header()

        while (not stop):
            sim_time = self.sim_timer.get_time()

            # Update the RTCs
            master_rtc.update(sim_time, evts)
            slave_rtc.update(sim_time, evts)

            # Try processing all events
            sync_transmitted = sync.tx(sim_time, master_rtc.get_time(), evts)
            dreq_transmitted = dreq.tx(sim_time, slave_rtc.get_time(), evts)
            sync_received    = sync.rx(sim_time, slave_rtc.get_time(),
                                       master_rtc.get_time())
            dreq_received    = dreq.rx(sim_time, master_rtc.get_time(),
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
                dreqresp.set_forward_delay(sync.seq_num,
                                           sync.one_way_delay)
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
                dreqresp.set_backward_delay(dreq.seq_num,
                                            dreq.one_way_delay)
                # Define true time offset and asymmetry
                dreqresp.set_truth(master_rtc.get_time(),
                                   slave_rtc.get_time())

                # Process all four timestamps
                results = dreqresp.process()

                # Estimate frequency offset
                y_est = freq_estimator.process(dreqresp.t1, dreqresp.t2)
                if (y_est is not None):
                    results["y_est"] = y_est

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
                stop = True

            self.check_progress(i_iter)

