# PTP Simulator

## Python Virtual Environment

If using *virtualenvwrapper*, run the following:

```
mkvirtualenv -r requirements.txt ptp-sim
```

## Running

For 10 message exchanges and debugging verbosity level:
```
python ptp_simulator.py -vvvvv -N 10
```

## Unit tests

From the root folder:
```
python -m unittest discover
```

## Test/demo scripts

* `ptp_plots.py`: Demonstrates several plots that can be generated using the
  `ptp.metrics` module.
* `ptp_estimators.py`: Demonstrates several estimators that can be used to
  post-process the PTP measurements and achieve better synchronization.
* `ls_vs_window.py` : Evaluates LS performance with respect to the observation
  window length.
* `psa_vs_window.py` : Evaluates PSA performance with respect to the observation
  window length.
* `kalman_demo.py`: Demonstrates how to evaluate Kalman filtering.
* `reader_demo.py`: Demonstrates how pre-acquired data can be loaded into the
  simulator.

### Running with acquired data

The script named `reader_demo.py` can load data acquired with real hardware and
post-process it using the simulator's algorithms. An example log file is
available at `data/example_log.json`. To process it, run:

```
python reader_demo.py -vvvvvv -f data/example_log.json -N 10
```

This will run 10 iterations only, and with debugging prints enabled.

To process it fully without log, run:

```
python reader_demo.py -f data/example_log.json
```
