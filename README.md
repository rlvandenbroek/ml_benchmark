Machine Learning (ML) benchmark based on the ML benchmarking paper [Beyond the Hype: Deep Neural Networks Outperform Established Methods Using A ChEMBL Bioactivity Benchmark Set (E.B. Lenselink, et al., 2017)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0232-0).

The dataset from the original paper (version) was used, which can be downloaded [here](https://data.4tu.nl/datasets/51a90ed0-9f8a-46fc-9497-d44aeced28ed/2).

## Usage:
1. From **datasetPrep.py** run *DatasetLenselink* and specify the location where you have downloaded the dataset. This script will retrieve the dataset containing only the relevant columns for replicating the original paper.
2. From **benchmark.py** run *benchmarkLenselink* to benchmark different ML methods. Currently supported methods are: Logistic Regression, Naïve Bayes, QLattice, Random Forest, and Support Vector Machine.

## Authors
[Remco L. van den Broek](https://github.com/rlvandenbroek)
