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

