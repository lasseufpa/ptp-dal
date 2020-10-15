# Batch Processing Recipes

This directory contains batch analysis recipes, i.e., JSON files describing the
analyses to be carried out based on collections of datasets. These recipes can
be useful, for example, to reproduce the results from publications.

For example, recipe `2020-ieee-access-draft.json` generates the results
contained in [1]. To run the analysis, run the batch processing tool as follows:

```
./batch.py -vvvv -a analyze recipes/2020-ieee-access-draft.json -j4
```

where argument `-j` controls the number of concurrent analysis jobs.

## References

[1] I. Freire et al., "Clock Synchronization Algorithms over PTP-Unaware
Networks: Reproducible Comparison Using an FPGA Testbed," submitted to the IEEE
Access.



