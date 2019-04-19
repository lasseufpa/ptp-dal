import unittest
from ptp.rtc import Rtc

class TestRtc(unittest.TestCase):
    def test_single_time_passage(self):
        """Test a single interval"""
        freq       = 125e6
        resolution = 0
        period_ns  = (1.0/freq) * 1e9
        rtc        = Rtc(freq, resolution)
        t_start    = rtc.get_time()

        # Advance time
        delta_ns   = 300
        delta      = delta_ns*1e-9
        rtc.update(delta)       # note time passed in units of second
        t_end      = rtc.get_time()

        # Results
        measured_delta = float(t_end - t_start)
        error          = abs(measured_delta - delta_ns)
        self.assertLess(error, period_ns)

    def test_mult_time_passage(self):
        """Test multiple intervals"""
        freq       = 125e6
        resolution = 0
        period_ns  = (1.0/freq) * 1e9
        rtc        = Rtc(freq, resolution)
        t_start    = rtc.get_time()

        # Advance time
        delta_ns   = 300
        delta      = delta_ns*1e-9
        n_deltas   = 3
        abs_time   = 0
        for i in range(0, n_deltas):
            abs_time += delta
            rtc.update(abs_time)
        t_end      = rtc.get_time()

        # Results
        measured_delta = float(t_end - t_start)
        error          = abs(measured_delta - (n_deltas * delta_ns))
        self.assertLess(error, period_ns)

    def test_freq_offset(self):
        """Test time-keeping with time-varying frequency offset"""
        freq       = 125e6
        resolution = 0
        period_ns  = (1.0/freq) * 1e9
        ppb        = 100
        rtc        = Rtc(freq, resolution, ppb)
        t_start    = rtc.get_time()

        # Advance time
        delta_ns   = 1e9
        delta      = delta_ns*1e-9
        rtc.update(delta)       # note time passed in units of second

        # Frequency offset during this interval:
        actual_freq_0  = rtc.freq_hz
        actual_ppb_0   = ((actual_freq_0 - freq)/freq)*1e9

        # Change freq. offset internally and advance time once again
        delta_ppb    = 5
        rtc.freq_hz *= (1 + (delta_ppb*1e-9))
        actual_ppb_1 = actual_ppb_0 + delta_ppb

        # Advance time once again
        rtc.update(delta)       # note time passed in units of second
        t_end      = rtc.get_time()

        # Results
        expected_delta = delta_ns + actual_ppb_0 + actual_ppb_1
        measured_delta = float(t_end - t_start)
        error          = abs(measured_delta - expected_delta)
        self.assertLess(error, period_ns)


if __name__ == '__main__':
    unittest.main()
