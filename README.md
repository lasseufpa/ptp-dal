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
* `kalman_test.py`: Demonstrates how to evaluate Kalman filtering.

