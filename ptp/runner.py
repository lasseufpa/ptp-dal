"""PTP simulation runner

Conventions:
- Simulation time is given in seconds
- RTC time is kept in seconds/nanoseconds
- Message periods are in seconds
- units are explicit within variable names where possible

"""
import logging, heapq
from ptp.rtc import *
from ptp.messages import *
from ptp.mechanisms import *
import ptp.dataset


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
                 rtc_clk_freq = 125e6, rtc_resolution = 0, target_ds = None):
        """PTP Runner class

        Args:
            n_iter         : Number of iterations
            sim_t_step     : Simulation time step in seconds
            sync_period    : Sync transmission period in seconds
            rtc_clk_freq   : RTC clock frequency in Hz
            rtc_resolution : RTC representation resolution in ns
            target_ds      : The target dataset
        """

        self.n_iter         = n_iter
        self.sync_period    = sync_period
        self.rtc_clk_freq   = rtc_clk_freq
        self.rtc_resolution = rtc_resolution
        self.target_ds      = target_ds

        # Simulation time
        self.sim_timer = SimTime(sim_t_step)

        # Simulation data
        self.data = list()

    def run(self):
        """Main loop

        Simulates PTP delay request-response message exchanges with
        corresponding time offset and delay estimations.

        """

        # Register the PTP message objects
        sync = PtpEvt("Sync", self.sync_period)
        dreq = PtpEvt("Delay_Req")

        # RTCs
        master_ppb_tol       = 0
        master_ppb_stability = 0
        slave_ppb_tol        = 60
        slave_ppb_stability  = 5
        master_rtc = Rtc(self.rtc_clk_freq, self.rtc_resolution, master_ppb_tol,
                         master_ppb_stability, "Master")
        slave_rtc = Rtc(self.rtc_clk_freq, self.rtc_resolution, slave_ppb_tol,
                        slave_ppb_stability, "Slave")

        # Main loop
        evts       = list()
        stop       = False
        i_iter     = 0
        dreqresps  = list()

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
                # Schedule the Delay_Req transmission
                dreq.sched_tx(sim_time, evts)

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
                # Process all four timestamps
                results = dreqresp.process(master_rtc.get_time(),
                                           slave_rtc.get_time())
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

    def generate_ds(self):
        """Post-process simulation data and generate dataset
        """

        if (self.target_ds is None):
            raise ValueError("Target dataset is missing")

        # Preallocate feature and label matrices
        feature_mtx = np.zeros(ptp.dataset.shape(self.target_ds,
                                                 self.n_iter)[0])
        label_mtx   = np.zeros(ptp.dataset.shape(self.target_ds,
                                                 self.n_iter)[1])

        for idx, results in enumerate(self.data):
            # Extract data to form dataset
            (feature_vec, label_vec) = ptp.dataset.features(results,
                                                            self.target_ds)
            feature_mtx[idx, :] = feature_vec
            label_mtx[idx, :]   = label_vec

        return(feature_mtx, label_mtx)

