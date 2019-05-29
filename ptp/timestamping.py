"""Timestamp definitions
"""

import numpy as np
import logging

class Timestamp():
    def __init__(self, sec_0 = 0, ns_0 = 0):
        """Timestamp type

        Represents time using separate counts of seconds and
        nanoseconds.

        Args:
        sec_0 : Initial seconds value
        ns_0  : Initial nanoseconds value

        """
        self.sec = int(sec_0)
        self.ns  = float(ns_0)

    def __add__(self, timestamp):
        """Sum another timestamp into self value

        This method supports adding both another Timestamp object as well as a
        float/int. When the argument is a float/int, it is assumed to be in
        nanoseconds.

        Args:
            timestamp : the other timestamp to add

        """

        if (isinstance(timestamp, Timestamp)):
            sec  = self.sec + timestamp.sec
            sec += np.floor((self.ns + timestamp.ns) / 1e9)
            ns   = np.mod((self.ns + timestamp.ns), 1e9)
        elif (isinstance(timestamp, float) or isinstance(timestamp, int)):
            sec  = self.sec
            sec += np.floor((self.ns + timestamp)/1e9);
            ns   = np.mod((self.ns + timestamp), 1e9)
        else:
            raise ValueError("Timestamp sum expects timestamp/float/int")

        assert(isinstance(self.sec, int))
        assert(isinstance(self.ns, float))
        assert(ns >= 0)
        assert(ns < 1e9)
        return Timestamp(sec, ns)

    def __sub__(self, timestamp):
        """Subtract another timestamp from self value

        Args:
            timestamp : the other timestamp to subtract
        """
        if (isinstance(timestamp, Timestamp)):
            sec  = self.sec - timestamp.sec
            sec += np.floor((self.ns - timestamp.ns)/1e9);
            ns   = np.mod((self.ns - timestamp.ns), 1e9)
        elif (isinstance(timestamp, float) or isinstance(timestamp, int)):
            sec  = self.sec
            sec += np.floor((self.ns - timestamp)/1e9);
            ns   = np.mod((self.ns - timestamp), 1e9)
        else:
            raise ValueError("Timestamp sum expects timestamp/float/int")

        # Protect from the issue of subtracting small number.
        # See https://docs.python.org/3/library/math.html#math.fmod
        if (ns == 1e9):
            logging.warning("Timestamp ns mod 1e9 resulted %e - set to 0" %(ns))
            sec += 1
            ns   = 0

        assert(isinstance(self.sec, int))
        assert(isinstance(self.ns, float))
        assert(ns >= 0)
        assert(ns < 1e9)
        return Timestamp(sec, ns)

    def __truediv__(self, other):
        """Divide timestamp"""
        ns  = (self.sec * 1e9) + self.ns
        ns /= other
        sec = np.floor(ns/1e9)
        ns  = ns % 1e9
        return Timestamp(sec, ns)

    def __str__(self):
        """Print sec and ns values"""
        return '{:5d} sec, {:9d} ns'.format(self.sec, int(np.floor(self.ns)))

    def __float__(self):
        """Cast timestamp to float"""
        return (float(self.sec) * 1e9) + self.ns

    def __abs__(self):
        """Abs of timestamp"""
        return abs((float(self.sec) * 1e9) + self.ns)

    def __int__(self):
        """Cast timestamp to int"""
        return int((self.sec * 1e9) + self.ns)


