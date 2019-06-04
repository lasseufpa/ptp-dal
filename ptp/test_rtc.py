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

    def test_known_freq_offset(self):
        """Test time-keeping with known time-varying frequency offset"""
        freq       = 125e6
        resolution = 0
        period_ns  = (1.0/freq) * 1e9
        tol_ppb    = 100
        rtc        = Rtc(freq, resolution, tol_ppb = tol_ppb)
        t_start    = rtc.get_time()
        t_sim      = 0

        # Advance time by 1 seconds
        delta_ns = 1e9
        delta    = delta_ns*1e-9
        t_sim   += delta

        rtc.update(t_sim)

        # Frequency offset during this interval:
        actual_freq_0  = rtc.freq_hz
        actual_ppb_0   = ((actual_freq_0 - freq)/freq)*1e9

        # Change freq. offset internally
        delta_ppb    = 5
        rtc.freq_hz *= (1 + (delta_ppb*1e-9))
        actual_ppb_1 = actual_ppb_0 + delta_ppb

        # Advance by 1 sec again
        t_sim += delta
        rtc.update(t_sim)

        # Results
        t_end          = rtc.get_time()
        measured_delta = float(t_end - t_start)
        expected_delta = 2*delta_ns + actual_ppb_0 + actual_ppb_1
        error          = abs(measured_delta - expected_delta)
        self.assertLess(error, period_ns)

    def test_random_freq_offset(self):
        """Test time-keeping with randomly time-varying frequency offset"""
        freq             = 125e6
        resolution       = 0
        period_ns        = (1.0/freq) * 1e9
        tol_ppb          = 100
        norm_var_freq_rw = 1e-16 # random-walk
        rtc              = Rtc(freq, resolution, tol_ppb = tol_ppb,
                               norm_var_freq_rw = norm_var_freq_rw)
        t_start          = rtc.get_time()
        t_sim            = 0

        # Advance time in steps of 1 second
        delta_ns = 1e9
        delta    = delta_ns*1e-9
        t_sim   += delta

        # Advance
        rtc.update(t_sim)

        # Frequency offset during this interval:
        actual_ppb_0 = rtc.get_freq_offset() * 1e9

        # Change freq. offset internally
        rtc._randomize_driving_clk(t_sim * 1e9)

        # Advance time again
        t_sim += delta
        rtc.update(t_sim)

        # Frequency offset during this interval:
        actual_ppb_1   = rtc.get_freq_offset() * 1e9

        # Results
        t_end          = rtc.get_time()
        measured_delta = float(t_end - t_start)
        expected_delta = 2*delta_ns + actual_ppb_0 + actual_ppb_1
        error          = abs(measured_delta - expected_delta)
        self.assertLess(error, period_ns)


if __name__ == '__main__':
    unittest.main()
