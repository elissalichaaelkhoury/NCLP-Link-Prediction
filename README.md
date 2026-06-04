# NCLP Link Prediction

This repository contains the implementation of the **Network Centrality Link Prediction (NCLP)** method optimized using a **Genetic Algorithm (GA)**.

The method combines centrality-based and common-neighbor-based measures to predict missing links in networks.

## Method Overview

NCLP uses the following structural measures:

- Betweenness centrality
- Closeness centrality
- Katz centrality
- Resource Allocation index
- Adamic-Adar index
- Common-neighbor centrality terms

The Genetic Algorithm optimizes six normalized weights corresponding to the centrality components of the two candidate nodes.

**Installation**

First, install the required libraries:

pip install -r requirements.txt

The required packages are:

networkx
numpy
scipy
scikit-learn
Dataset Format

The code supports edge-list datasets.

*Unweighted space-separated format*

source target
1 2
2 3
3 4

*Weighted space-separated format*

source target weight
1 2 1
2 3 4
3 4 2

*Comma-separated format*

source,target
1,2
2,3
3,4

**Usage**

*Unweighted space-separated dataset*

python run_pipeline.py --file_path "path/to/dataset.txt" --delimiter space

*Weighted space-separated dataset*
python run_pipeline.py --file_path "path/to/dataset.txt" --delimiter space --weighted

*Comma-separated dataset*
python run_pipeline.py --file_path "path/to/dataset.edges" --delimiter comma

Example
python run_pipeline.py --file_path "data/karate.txt" --delimiter space --runs 10
