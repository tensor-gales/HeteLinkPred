# On the Impact of Feature Heterophily on Link Prediction with Graph Neural Networks
This repository contains the code, script and data for NeurIPS 2024 submission "On the Impact of Feature Heterophily on Link Prediction with Graph Neural Networks"

## Run Experiments

To install the dependecies in a new conda environment, run
```
$ conda create --name <env> --file hetelinkpred/conda_env/dgl.txt
```
Scripts used to get results for BUDDY can be found in /subgraph-sketching/scripts. \
Other scripts can be found in /hetelinkpred/shell-scripts-hetelinkpred.

## GNN Encoders Supported
- GraphSAGE
- GCN
- BUDDY 

## Decoders Supported
- MLP
- DistMult
- Dot product

## Heuristics Supported 
- Common Neighbor
- Adamic Adar
- Resource Allocation
- Personalized Page Rank

## Datasets
- Synthetic dataset of varying heterophily level
- ogbl-collab
- ogbl-citation2
- E-Commerce

Please following the script below to unzip the synthetic and e-commerce datasets.
```bash
# sudo apt-get install p7zip-full # install 7z for ubuntu.
# brew install p7zip # install 7z for mac
# For windows, download 7z from https://www.7-zip.org/download.html

cd hetelinkped
7z x dataset.7z.part.001
7z x dataset.7z
```
