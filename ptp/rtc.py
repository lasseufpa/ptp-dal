"""Real-time clock definitions
"""

import random, logging
from .timestamping import *

class Rtc():
    def __init__(self, clk_freq_hz, resolution_ns, label):
        """Real-time Clock (RTC)

        Args:
        clk_freq_hz   : Frequency (in Hz) of the driving clock signal
        resolution_ns : Timestamp resolution in nanoseconds
        label         : RTC label
        """

        # Start the rtc with a random time and phase
        sec_0 = random.randint(0, 1000)
        ns_0  = random.uniform(0, 1e9)

        # Nominal increment value in nanoseconds
        inc_val_ns = (1.0/clk_freq_hz)*1e9

        # The phase is the instant within the period of the driving clock signal
        # where the rising edge is located
        phase_0_ns = random.uniform(0, inc_val_ns)

        self.inc_cnt    = 0
        self.freq_hz    = clk_freq_hz  # driving clock signal freq.
        self.period_ns  = inc_val_ns   # driving clock signal period
        self.inc_val_ns = inc_val_ns   # increment value
        self.phase_ns   = phase_0_ns   # phase
        self.time       = Timestamp(sec_0, ns_0)
        self.toffset    = Timestamp()
        self.label      = label

        logger = logging.getLogger('Rtc')
        logger.debug("Initialized the %s RTC" %(self.label))
        logger.debug("%-16s\t %f ns" %("Increment value:", self.inc_val_ns))
        logger.debug("%-16s\t %f ns" %("Initial phase:", self.phase_ns))
        logger.debug("%-16s\t %s" %("Initial time:", self.time))

    def update(self, t_sim):
        """Update the RTC time

        Args:
            t_sim : simulation time in seconds
        """

        t_sim_ns = t_sim * 1e9

        # Check how many times the RTC has incremented so far:
        n_incs = math.floor((t_sim_ns - self.phase_ns) / (self.period_ns))

        # Prevent negative number of increments
        if (n_incs < 0):
            n_incs = 0

        # Track the number of increments that haven't been taken into account
        # yet
        new_incs = n_incs - self.inc_cnt

        # Elapsed time at the RTC since last update:
        elapsed_ns = new_incs * self.inc_val_ns
        # NOTE: the elapsed time depends on the increment value that is
        # currently configured at the RTC. The number of increments, in
        # contrast, does not depend on the current RTC configuration.

        # Update the increment counter
        self.inc_cnt = n_incs

        # Update the RTC seconds count:
        self.time += elapsed_ns

        logger = logging.getLogger('Rtc')
        logger.debug("[%-6s] Simulation time: %f ns" %(self.label, t_sim_ns))
        logger.debug("[%-6s] Advance RTC by %u ns" %(self.label, elapsed_ns))
        logger.debug("[%-6s] New RTC time: %s" %(self.label, self.time))

    def get_time(self):
        """Get current RTC time
        """
        return self.time
