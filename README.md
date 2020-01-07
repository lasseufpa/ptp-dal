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

## Test/demo/helper scripts

* `ptp_plots.py`: Demonstrates several plots that can be generated using the
  `ptp.metrics` module.
* `ptp_estimators.py`: Demonstrates several estimators that can be used to
  post-process the PTP measurements and achieve better synchronization.
* `window_optimizer_demo.py` : Evaluates performance of estimators with respect
  to the observation window length.
* `kalman_demo.py`: Demonstrates how to evaluate Kalman filtering.
* `reader_demo.py`: Demonstrates how pre-acquired data can be loaded into the
  simulator.
* `capture.py` : Acquire timestamp data from testbed
* `catalog.py` : Catalogs datasets acquired from testbed

## Acquire Data from the Testbed

Run:
```
./capture.py -vvvvv
```

where `-vvvvv` controls the verbosity level.

## Running with acquired data

First download a dataset that was acquired with the testbed. They are kept at
`Lasse100` machine. Provided that you have SSH access to this machine, you can
download a JSON dataset (such as `serial-20200107-111932.json`) as follows:

```
./download.py serial-20200107-111932.json
```

Then run `reader_demo.py` with the downloaded dataset. This script will
post-process the timestamps using all implemented algorithms.

```
./reader_demo.py -vvvv -f data/serial-20200107-111932.json
```

After that, a set of results will be available within the `plots/` directory.

An example log file is also available locally within the `data/` folder. You can
test it with:

```
python reader_demo.py -vvvvvv -f data/example_log.json -N 10
```

This will run 10 iterations only, and with debugging prints enabled.

## Catalog datasets

Assuming for instance that the datasets are located at the `data/` folder
(default), run the following to catalog them based on their metadata:

```
./catalog.py -d data/
```

