"""Real-time clock definitions
"""

import random, logging, math
from ptp.timestamping import Timestamp


class Rtc():
    def __init__(self, nom_freq_hz, resolution_ns, ppb = 0, label="RTC"):
        """Real-time Clock (RTC)

        Args:
        nom_freq_hz   : Nominal frequency (in Hz) of the driving clock signal
        resolution_ns : Timestamp resolution in nanoseconds
        ppb           : Frequency tolerance in ppb
        label         : RTC label
        """

        # Start the rtc with a random time and phase
        sec_0 = random.randint(0, 0)
        ns_0  = random.uniform(0, 5e3)

        # Nominal increment value in nanoseconds
        inc_val_ns = (1.0/nom_freq_hz)*1e9

        # Actual initial driving frequency, considering the initial fractional
        # freq. offset of the driving clock
        freq_offset_0_ppb = random.uniform(-ppb, ppb)
        freq_offset_0     = freq_offset_0_ppb * 1e-9
        freq_hz           = nom_freq_hz * (1 + freq_offset_0)

        # The phase is the instant within the period of the driving clock signal
        # where the rising edge is located
        phase_0_ns = random.uniform(0, inc_val_ns)

        self.inc_cnt    = 0
        self.freq_hz    = freq_hz       # driving clock signal freq.
        self.inc_val_ns = inc_val_ns   # increment value
        self.phase_ns   = phase_0_ns   # phase
        self.time       = Timestamp(sec_0, ns_0)
        self.toffset    = Timestamp()
        self.label      = label
        self.t_last_inc = 0

        logger = logging.getLogger('Rtc')
        logger.debug("Initialized the %s RTC" %(self.label))
        logger.debug("%-16s\t %f ns" %("Increment value:", self.inc_val_ns))
        logger.debug("%-16s\t %f ns" %("Initial phase:", self.phase_ns))
        logger.debug("%-16s\t Freq: %f MHz\tPeriod %f ns" %(
            "Driving clock", self.freq_hz/1e6, 1.0/self.freq_hz))
        logger.debug("%-16s\t %s" %("Initial time:", self.time))

    def update(self, t_sim):
        """Update the RTC time

        Args:
            t_sim : absolute simulation time in seconds
        """

        t_sim_ns = t_sim * 1e9

        # Based on the current RTC driving clock period (which changes over
        # time), check how many times the RTC has incremented since last time
        rtc_period_ns = (1.0/self.freq_hz) * 1e9
        n_new_incs = math.floor((t_sim_ns - self.t_last_inc) / (rtc_period_ns))
        # TODO: model phase noise in addition to freq. noise

        # Prevent negative number of increments
        assert(n_new_incs >= 0)

        # Elapsed time according to the RTC since last update:
        elapsed_ns = n_new_incs * self.inc_val_ns
        # NOTE: the elapsed time depends on the increment value that is
        # currently configured at the RTC. The number of increments, in
        # contrast, depends only on the actual period of the driving clock.

        # Update:
        self.inc_cnt    += n_new_incs    # increment counter
        self.time       += elapsed_ns    # RTC tim
        self.t_last_inc += (n_new_incs * rtc_period_ns)

        logger = logging.getLogger('Rtc')
        logger.debug("[%-6s] Simulation time: %f ns" %(self.label, t_sim_ns))
        logger.debug("[%-6s] Advance RTC by %u ns" %(self.label, elapsed_ns))
        logger.debug("[%-6s] New RTC time: %s" %(self.label, self.time))

    def get_time(self):
        """Get current RTC time
        """
        return self.time
