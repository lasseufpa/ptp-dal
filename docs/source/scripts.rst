PTP-DAL Programs
=======================================

Main programs
~~~~~~~~~~~~~~~~~~~~~~~~

-  ``analyze.py`` : Analyzes a dataset and compares synchronization
   algorithms.
-  ``batch.py`` : Runs a batch of analyses (see the batch processing
   `recipes <recipes/>`__).
-  ``catalog.py`` : Catalogs datasets acquired with the testbed.
-  ``dataset.py`` : Downloads and searches datasets by communicating
   with the dataset database.

Complementary programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``compress.py`` : Compresses a given dataset captured with the
   testbed.
-  ``ptp_plots.py``: Demonstrates a few plots that can be generated
   using the ``ptp.metrics`` module.
-  ``ptp_estimators.py``: Demonstrates estimators that can be used to
   post-process PTP measurements.
-  ``simulate.py`` : Simulates PTP clocks and generates a timestamp
   dataset that can be processed with the same scripts used to process a
   testbed-generated dataset.
-  ``window_optimizer_demo.py`` : Evaluates the performance of
   window-based estimators according to the observation window length.
-  ``kalman_demo.py``: Demonstrates the evaluation of Kalman filtering.