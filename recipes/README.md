# Batch Processing Recipes

This directory contains batch analysis recipes, i.e., JSON files describing the
analyses to be carried out based on collections of datasets. These recipes can
be useful, for example, to reproduce the results from publications.

For example, recipe `2021-ieee-access-paper.json` generates the results
contained in [\[1\]](https://ieeexplore.ieee.org/document/9334990). To run the
analysis, run the batch processing tool as follows:

```
./batch.py -vvvv recipes/2021-ieee-access-paper.json -j4
```

where argument `-j` controls the number of concurrent analysis jobs.

## References

[1] I. Freire, C. Novaes, I. Almeida, E. Medeiros, M. Berg and A. Klautau,
"Clock Synchronization Algorithms Over PTP-Unaware Networks: Reproducible
Comparison Using an FPGA Testbed," in IEEE Access, vol. 9, pp. 20575-20601,
2021, doi: 10.1109/ACCESS.2021.3054164 [(available online on IEEE
Xplore)](https://ieeexplore.ieee.org/document/9334990).



