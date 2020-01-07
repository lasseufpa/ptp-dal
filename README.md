# PTP Research Framework

This repository contains a Python package and associated scripts/applications
for analysis of PTP synchronization algorithms, based on timestamps that are
acquired with real-hardware using our FPGA-based testbed. The testbed captures
raw timestamps into JSON datasets. This framework loads these datasets and
generates several estimations based on them for offline analysis.

The main motivation for this approach is that it allows the operation of several
different algorithms on the same data. In the end, this allows a fair comparison
between the results of the different algorithms. While in practice these
algorithms could be computed in real-time by a PTP node, most likely only one of
them would run at a time to discipline the node's real-time clock (RTC).

## Python Virtual Environment

If using *virtualenvwrapper*, run the following:

```
mkvirtualenv -r requirements.txt ptp
```

## Unit tests

From the root folder:
```
python -m unittest discover
```

## Test/demo/helper scripts

Main scripts:
* `analyze.py` : Analyzes testbed-acquired data and compares synchronization
  algorithms.
* `capture.py` : Acquires timestamp data from the testbed in real-time.
* `catalog.py` : Catalogs datasets acquired with the testbed.
* `download.py` : Downloads a testbed dataset from our local repository (kept at
  `Lasse100` machine).

Complementary:
* `ptp_plots.py`: Demonstrates several plots that can be generated using the
  `ptp.metrics` module.
* `ptp_estimators.py`: Demonstrates several estimators that can be used to
  post-process the PTP measurements and achieve better synchronization.
* `ptp_simulator.py` : Simulates PTP clocks and generates a timestamp dataset
  that can be processed with the same scripts that process a testbed dataset.
* `window_optimizer_demo.py` : Evaluates performance of estimators with respect
  to the observation window length.
* `kalman_demo.py`: Demonstrates how to evaluate Kalman filtering.

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

Then run `analyze.py` with the downloaded dataset. This script will
post-process the timestamps using all implemented algorithms.

```
./analyze.py -vvvv -f data/serial-20200107-111932.json
```

After that, a set of results will be available within the `plots/` directory.

An example log file is also available locally within the `data/` folder. You can
test it with:

```
python analyze.py -vvvvvv -f data/example_log.json -N 10
```

This will run 10 iterations only, and with debugging prints enabled.

## Dataset Cataloging

Assuming for instance that the datasets to be cataloged are located within the
`data/` folder (default), run the following:

```
./catalog.py -d data/
```

This will then create a `README.html` file in the target directory (`data/` in
the example) containing the metadata of all cataloged datasets.

## Timestamp Simulator

This application can generate a timestamp dataset that is formatted similarly to
the datasets that are acquired from the testbed. Consequently, the same scripts
that process testbed datasets can process the simulated dataset.

To generate a dataset, define the target number of message exchanges and run
with command-line argument `--save`. For example, for 1000 exchanges and
debugging verbosity level, run:

```
./ptp_simulator.py -vvvvv -N 1000 --save
```

In the end, the resulting dataset is placed in `data/`.

> NOTE: all datasets generated via simulation are named with `runner-`
> prefix. In contrast, datasets acquired serially from the testbed are named
> with `serial-` prefix.
