"""Real-time clock definitions
"""

import heapq
import logging
import math
import random

from ptp.timestamping import Timestamp


class Rtc():
    """Real-time Clock (RTC) Model

    The model incorporates random-walk frequency noise and random-walk time
    noise. These are controlled by the normalized variance of the frequency
    offset random-walk iid step (norm_var_freq_rw) and the normalized variance
    of the time offset random-walk (norm_var_time_rw). Importantly, these
    normalized variances are ultimately multiplied by the update period of the
    model as suggested in [1].

    References:

    1. G. Giorgi and C. Narduzzi, "Performance Analysis of Kalman-Filter-Based
       Clock Synchronization in IEEE 1588 Networks," in IEEE Transactions on
       Instrumentation and Measurement, vol. 60, no. 8, pp. 2902-2909, Aug.
       2011.


    Args:
        nom_freq_hz      : Nominal frequency (Hz) of the driving clock
        resolution_ns    : Timestamp resolution in nanoseconds
        tol_ppb          : Frequency tolerance in ppb
        norm_var_freq_rw : Freq. offset random-walk's normalized variance
        norm_var_time_rw : Time offset random-walk's normalized variance
        label            : RTC label
        ts_quantization  : Enables quantization of the time scale

    """
    def __init__(self,
                 nom_freq_hz,
                 resolution_ns,
                 tol_ppb=0.0,
                 norm_var_freq_rw=0.0,
                 norm_var_time_rw=0.0,
                 label="RTC",
                 ts_quantization=True):
        # Time scale quantization
        self.ts_quantization = ts_quantization

        # Start the rtc with a random time and phase
        sec_0 = random.randint(0, 0)
        ns_0 = random.uniform(0, 5e3)

        # Nominal increment value in nanoseconds
        inc_val_ns = (1.0 / nom_freq_hz) * 1e9
        # TODO: a practical RTC cannot represent any arbitrary increment
        # value. It will be limited by its fixed-point resolution. Use
        # `resolution_ns` here.

        # Actual initial driving frequency, considering the initial fractional
        # freq. offset of the driving clock due to "manufacture tolerance"
        freq_offset_0_ppb = random.uniform(-tol_ppb, tol_ppb)
        freq_offset_0 = freq_offset_0_ppb * 1e-9
        freq_hz = nom_freq_hz * (1 + freq_offset_0)

        # The phase is the instant within the period of the driving clock
        # signal where the rising edge is located
        phase_0_ns = random.uniform(0, inc_val_ns)

        # Constants
        self._nom_freq_hz = nom_freq_hz  # Nominal driving clock freq.
        self.label = label

        # Variable over time:
        self.inc_cnt = 0
        self.freq_hz = freq_hz  # Current driving clock signal freq.
        self.inc_val_ns = inc_val_ns  # increment value
        self.phase_ns = phase_0_ns  # phase
        self.time = Timestamp(sec_0, ns_0)
        self.toffset = Timestamp()
        self.t_last_inc = 0

        # Clock modeling
        self._model_update_period_ns = 1e7
        self._model_sdev_freq_rw = math.sqrt(1e-9 * norm_var_freq_rw *
                                             self._model_update_period_ns)
        self._model_sdev_time_rw = math.sqrt(1e-9 * norm_var_time_rw *
                                             self._model_update_period_ns)
        self._model_t_last_update = 0

        logger = logging.getLogger('Rtc')
        logger.debug("Initialized the %s RTC" % (self.label))
        logger.debug("%-16s\t %f ns" % ("Increment value:", self.inc_val_ns))
        logger.debug("%-16s\t %f ns" % ("Initial phase:", self.phase_ns))
        logger.debug("%-16s\t Freq: %f MHz\tPeriod %f ns" %
                     ("Driving clock", self.freq_hz / 1e6, 1.0 / self.freq_hz))
        logger.debug("%-16s\t %s" % ("Initial time:", self.time))

    def _randomize_driving_clk(self, t_sim_ns):
        """Update the properties of the driving clock

        C.f. model description in the constructor method.

        Args:
            t_sim_ns : Simulation time in ns

        Returns:
            True when udpated
        """

        t_next_update = self._model_t_last_update + \
            self._model_update_period_ns

        # Is it time to update the frequency/phase?
        if (t_sim_ns >= t_next_update):
            # Random-walk frequency noise
            noise_y = random.gauss(0, self._model_sdev_freq_rw)
            # The above is normalized - scale back to Hz
            self.freq_hz += (self._nom_freq_hz * noise_y)

            # Random-walk time offset noise due to phase noise
            noise_x = random.gauss(0, self._model_sdev_time_rw)
            self.time += noise_x

            # Save the update time
            self._model_t_last_update = t_sim_ns

            logger = logging.getLogger('Rtc')
            logger.debug("[%-6s] New driving freq: %f MHz" %
                         (self.label, self.freq_hz / 1e6))

            return True

    def update(self, t_sim, evts=None):
        """Update the RTC time

        When the event heap queue is passed by argument, it is assumed that the
        caller controls events using this queue. In this case, this function
        always schedules the next update to the RTC driving clock frequency.

        Args:
            t_sim : absolute simulation time in seconds
            evts  : Event heap queue

        """

        t_sim_ns = t_sim * 1e9

        # Simulate driving clock frequency.
        #
        # Schedule periodic wake-ups for the RTC in order to simulate its
        # driving frequency randomly changing over time
        if (evts is not None and self._model_sdev_freq_rw != 0):
            if (self._randomize_driving_clk(t_sim_ns)):
                # Schedule next update
                heapq.heappush(evts,
                               (t_sim + (self._model_update_period_ns * 1e-9)))

        # Based on the current RTC driving clock period (which changes over
        # time), check how many times the RTC has incremented since last time
        rtc_period_ns = (1.0 / self.freq_hz) * 1e9

        # The number of increments controls the time scale behavior. If it is
        # truncated to integer values, the time-scale will be quantized, as in
        # real RTC hardware.
        n_new_incs = (t_sim_ns - self.t_last_inc) / (rtc_period_ns)
        if (self.ts_quantization):
            n_new_incs = math.floor(n_new_incs)
        # TODO: model phase noise in addition to freq. noise

        # Prevent negative number of increments
        assert (n_new_incs >= 0)

        # Elapsed time according to the RTC since last update:
        elapsed_ns = n_new_incs * self.inc_val_ns
        # NOTE: the elapsed time depends on the increment value that is
        # currently configured at the RTC. The number of increments, in
        # contrast, depends only on the actual period of the driving clock.

        # Update:
        self.inc_cnt += n_new_incs  # increment counter
        self.time += elapsed_ns  # RTC tim
        self.t_last_inc += (n_new_incs * rtc_period_ns)

        logger = logging.getLogger('Rtc')
        logger.debug("[%-6s] Simulation time: %f ns" % (self.label, t_sim_ns))
        logger.debug("[%-6s] Advance RTC by %u ns" % (self.label, elapsed_ns))
        logger.debug("[%-6s] New RTC time: %s" % (self.label, self.time))

    def get_time(self):
        """Get current RTC time
        """
        return self.time

    def get_freq_offset(self):
        """Get the current fractional (normalized) frequency offset"""
        return ((self.freq_hz - self._nom_freq_hz) / self._nom_freq_hz)
