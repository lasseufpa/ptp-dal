"""Timestamp definitions
"""

import math

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
            sec += math.floor((self.ns + timestamp.ns)/1e9);
            ns   = (self.ns + timestamp.ns) % 1e9
        elif (isinstance(timestamp, float) or isinstance(timestamp, int)):
            sec  = self.sec
            sec += math.floor((self.ns + timestamp)/1e9);
            ns   = (self.ns + timestamp) % 1e9
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
            sec += math.floor((self.ns - timestamp.ns)/1e9);
            ns   = (self.ns - timestamp.ns) % 1e9
        elif (isinstance(timestamp, float) or isinstance(timestamp, int)):
            sec  = self.sec
            sec += math.floor((self.ns - timestamp)/1e9);
            ns   = (self.ns - timestamp) % 1e9
        else:
            raise ValueError("Timestamp sum expects timestamp/float/int")

        if (sec < 0):
            sec = 0
            ns  = 0

        assert(isinstance(self.sec, int))
        assert(isinstance(self.ns, float))
        assert(ns >= 0)
        assert(ns < 1e9)
        return Timestamp(sec, ns)

    def __str__(self):
        """Print sec and ns values"""
        return '{} sec, {:9} ns'.format(self.sec, math.floor(self.ns))


