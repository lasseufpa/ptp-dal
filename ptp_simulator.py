#!/usr/bin/env python

"""PTP Simulator

Conventions:
- Simulation time is given in seconds
- RTC time is kept in seconds/nanoseconds
- Message periods are in seconds
- units are explicit within variable names where possible

"""
import argparse, logging, sys, math, heapq
from ptp.rtc import *
from ptp.messages import *
from ptp.mechanisms import *


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


def run(n_iter, sim_t_step):
    """Main loop

    Args:
        n_iter        : Number of iterations
        sim_t_step    : Simulation time step in seconds
    """

    # Constants
    sync_period    = 1.0/16 # in seconds
    rtc_clk_freq   = 125e6  # in Hz
    rtc_resolution = 0 # TODO

    # Register the PTP message objects
    sync = PtpEvt("Sync", sync_period)
    dreq = PtpEvt("Delay_Req")

    # RTCs
    master_rtc = Rtc(rtc_clk_freq, rtc_resolution, "Master")
    slave_rtc  = Rtc(rtc_clk_freq, rtc_resolution, "Slave")

    # Simulation time
    sim_timer = SimTime(sim_t_step)

    # Main loop
    evts       = list()
    stop       = False
    i_msg      = 0
    dreqresps  = list()

    # Start with a sync transmission
    sync.next_tx = 0

    DelayReqResp(0,0).log_header()

    while (not stop):
        sim_time = sim_timer.get_time()

        # Update the RTCs
        master_rtc.update(sim_time)
        slave_rtc.update(sim_time)

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
            dreqresp.process(master_rtc.get_time(),
                             slave_rtc.get_time())
            #TODO: include ground truth on processing step
            # Message exchange count
            i_msg += 1

        # Update simulation time
        if (len(evts) > 0):
            next_evt = heapq.heappop(evts)
            sim_timer.advance(next_evt)
        else:
            sim_timer.step()

        # Stop criterion
        if (i_msg >= n_iter):
            stop = True


def main():
    parser = argparse.ArgumentParser(description="PTP Simulator")
    parser.add_argument('-N', '--num-iter',
                        default=10,
                        type=int,
                        help='Number of iterations.')
    parser.add_argument('-t', '--sim-step',
                        default=1e-9,
                        type=float,
                        help='Simulation time step in seconds.')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Verbosity (logging) level")
    args     = parser.parse_args()
    args     = parser.parse_args()
    num_iter = args.num_iter
    sim_step = args.sim_step
    verbose  = args.verbose

    logging_level = 70 - (10 * verbose) if verbose > 0 else 0
    logging.basicConfig(stream=sys.stderr, level=logging_level)

    run(num_iter, sim_step)

if __name__ == "__main__":
    main()
