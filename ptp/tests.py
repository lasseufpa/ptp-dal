import unittest
from timestamping import *

class TestTimestamping(unittest.TestCase):

    def test_sum(self):
        """Summation with no ns wrapping"""
        x = Timestamp(1, 200)
        y = Timestamp(3, 120)
        z = x + y

        # Check that x and y are preserved
        self.assertEqual(x.sec, 1)
        self.assertEqual(x.ns, 200)
        self.assertEqual(y.sec, 3)
        self.assertEqual(y.ns, 120)

        # Check result of sum
        self.assertEqual(z.sec, 4)
        self.assertEqual(z.ns, 320)

    def test_sub(self):
        """Subtraction with no ns wrapping"""
        x = Timestamp(1, 100)
        y = Timestamp(3, 120)
        z = y - x

        # Check that x and y are preserved
        self.assertEqual(x.sec, 1)
        self.assertEqual(x.ns, 100)
        self.assertEqual(y.sec, 3)
        self.assertEqual(y.ns, 120)

        # Check result of sum
        self.assertEqual(z.sec, 2)
        self.assertEqual(z.ns, 20)

    def test_wrap_positive(self):
        """Summation with positive ns wrapping"""
        x = Timestamp(1, 999999980)
        y = Timestamp(3, 120)
        z = x + y
        self.assertEqual(z.sec, 5)
        self.assertEqual(z.ns, 100)

    def test_wrap_negative(self):
        """Subtraction with negative ns wrapping"""
        x = Timestamp(1, 200)
        y = Timestamp(3, 120)
        z = y - x
        self.assertEqual(z.sec, 1)
        self.assertEqual(z.ns, 999999920)

    def test_positiveness(self):
        """Subtraction where total result would become negative"""
        x = Timestamp(1, 200)
        y = Timestamp(3, 120)
        z = x - y
        self.assertEqual(z.sec, 0)
        self.assertEqual(z.ns, 0)

    def test_add_float(self):
        """Add float with no ns wrapping"""
        x = Timestamp(1, 200)
        y = 120.0
        z = x + y

        # Check result of sum
        self.assertEqual(z.sec, 1)
        self.assertEqual(z.ns, 320)

    def test_add_float_wrap(self):
        """Add float with wrapping"""
        x = Timestamp(1, 999999900)
        y = 120.0
        z = x + y

        # Check result of sum
        self.assertEqual(z.sec, 2)
        self.assertEqual(z.ns, 20)

    def test_add_int(self):
        """Add int with no ns wrapping"""
        x = Timestamp(1, 200)
        y = 120
        z = x + y

        # Check result of sum
        self.assertEqual(z.sec, 1)
        self.assertEqual(z.ns, 320)

    def test_add_float_wrap(self):
        """Add int with wrapping"""
        x = Timestamp(1, 999999900)
        y = 120
        z = x + y

        # Check result of sum
        self.assertEqual(z.sec, 2)
        self.assertEqual(z.ns, 20)

    def test_sub_float(self):
        """Add float with no ns wrapping"""
        x = Timestamp(1, 200)
        y = 120.0
        z = x - y

        # Check result of sum
        self.assertEqual(z.sec, 1)
        self.assertEqual(z.ns, 80)

    def test_sub_float_wrap(self):
        """Add float with wrapping"""
        x = Timestamp(1, 200)
        y = 220.0
        z = x - y

        # Check result of sum
        self.assertEqual(z.sec, 0)
        self.assertEqual(z.ns, 999999980)

    def test_sub_int(self):
        """Add int with no ns wrapping"""
        x = Timestamp(1, 200)
        y = 120
        z = x - y

        # Check result of sum
        self.assertEqual(z.sec, 1)
        self.assertEqual(z.ns, 80)

    def test_sub_int_wrap(self):
        """Add int with wrapping"""
        x = Timestamp(1, 200)
        y = 220
        z = x - y

        # Check result of sum
        self.assertEqual(z.sec, 0)
        self.assertEqual(z.ns, 999999980)

    def test_float_cast(self):
        """Cast timestamp into float"""
        x = Timestamp(1, 200.2)
        z = float(x)

        self.assertEqual(z, 1000000200.2)

    def test_int_cast(self):
        """Cast timestamp into int"""
        x = Timestamp(1, 200.2)
        z = int(x)

        self.assertEqual(z, 1000000200)

if __name__ == '__main__':
    unittest.main()
