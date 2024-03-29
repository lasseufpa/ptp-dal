Overview
==========

The PTP-DAL project consists of a Python package and scripts to investigate
synchronization algorithms applied on top of the IEEE 1588 precision time
protocol (PTP). The project focuses on offline analysis by processing datasets
of timestamps collected from real hardware. Using this strategy, the user can
process the same dataset with varying parameters and algorithms until achieving
the best synchronization performance.

The PTP-DAL library implements several algorithms, such as packet
selection, least-squares, and Kalman filtering. These are applied
independently on the timestamps provided by a given dataset. This
approach is analogous to running several algorithms in parallel in a
real-time implementation.

After processing the selected algorithms, PTP-DAL outputs a
comprehensive set of results comparing the synchronization performance
achieved by each algorithm, with timing metrics such as the maximum
absolute time error (max\|TE\|), maximum time interval error (MTIE), and
so on. Additionally, the results include analyses of several aspects of
the PTP network and the surrounding environment, such as the packet
delay variation (PDV), PTP delay distributions, and temperature
variations.

The project was specifically developed to analyze datasets of timestamps
generated by the FPGA-based PTP synchronization testbed developed by
`LASSE - 5G & IoT Research Group <https://www.lasse.ufpa.br/>`__. This
testbed has been detailed in various publications, including:

1. `"Clock Synchronization Algorithms Over PTP-Unaware Networks:
   Reproducible Comparison Using an FPGA Testbed," in IEEE Access,
   2021 <https://ieeexplore.ieee.org/document/9334990>`__.
2. `"5G Fronthaul Synchronization via IEEE 1588 Precision Time Protocol:
   Algorithms and Use Cases," Ph.D. thesis, Federal University of Pará, Dec.
   2020.
   <https://igorfreire-personal-page.s3.amazonaws.com/publications/2020_phd_thesis_igor_freire.pdf>`__
3. `"Testbed Evaluation of Distributed Radio Timing Alignment Over
   Ethernet Fronthaul Networks," in IEEE Access, 2020.
   <https://ieeexplore.ieee.org/document/9088987>`__
4. `"An FPGA-based Design of a Packetized Fronthaul Testbed with IEEE
   1588 Clock Synchronization," European Wireless 2017.
   <https://ieeexplore.ieee.org/document/8011327>`__

In particular, Chapter 4 from reference
`[2] <https://igorfreire-personal-page.s3.amazonaws.com/publications/2020_phd_thesis_igor_freire.pdf>`__
provides the most comprehensive description of the testbed and the
dataset acquisition process, while Chapter 3 covers the algorithms
supported by the PTP-DAL project.

The adopted datasets of timestamps comprise a large number of PTP
two-way exchanges. Each exchange corresponds to a row in the dataset and
includes a set of timestamps. More specifically, each row consists of
the four timestamps involved in the two-way PTP packet exchange (t1, t2,
t3, and t4), as well as auxiliary timestamps. The auxiliary timestamps
indicate the actual one-way delay of each PTP packet and the true time
offset affecting the slave at that moment. Ultimately, this supplemental
information allows for analyzing the error between each time offset
estimator and the actual time offset experienced by the slave clock at
any point in time.

The datasets produced by the testbed can be made available on demand. If you
are interested in exploring PTP-DAL using datasets acquired from LASSE's PTP
synchronization testbed, please read the dataset access section and contact us
directly over `email <mailto:ptp.dal@gmail.com>`__. Otherwise, this repository
contains a simulator capable of generating compatible datasets through
simulation.