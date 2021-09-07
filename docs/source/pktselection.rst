Packet Selection
=======================================

Implements packet selection with a selection operator of choice. The following
operators are supported: sample-average, sample-median, sample-minimum,
sample-maximum, sample-mode, and exponentially-weighted moving average (EWMA).

.. autoclass:: ptp.pktselection.PktSelection
   :members: