# Graph Positional Autoencoders as Self-supervised Learners (GraphPAE)

This is the Pytorch implementation for ["Graph Positional Autoencoders as Self-supervised Learners"]().

![](https://github.com/liuyang-tian/GraphPAE/blob/main/GraphPAE.png)

## Requirements
```
accelerate==0.33.0
numpy==1.24.4
ogb==1.3.6
pandas==2.2.3
PyYAML==6.0.2
rdkit_pypi==2022.9.5
scikit_learn==1.3.2
scipy==1.8.1
torch==2.1.2+cu121
torch_geometric==2.6.1
torch_scatter==2.1.2+pt21cu121
tqdm==4.66.5
```

## Node Classification

### Download Datasets
If not otherwise specified, the code will automatically download the required datasets during data preprocessing.

For BlogCatalog, please unzip `./dataset/blog.zip` before running the program.

For Penn94, We use the datasets provided in ["Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods"]([https://arxiv.org/abs/2110.14446](https://arxiv.org/abs/2110.14446)). It is available on [Non-Homophily-Large-Scale](https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/data/facebook100/Penn94.mat). Download the file and put it under `./dataset`. 

For arXiv-year and Penn94, we adopt the same data splits as provided in the [Non-Homophily-Large-Scale](https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master/data/splits). They have been saved in `./dataset`.

### Running
(1) To preprocess the datasets, move into `./node` or `./node_batch` and run `preprocess.py`.

(2) Train and evaluate GraphPAE for node classification by running `train_node.py` or `train_batch.py`.

## Graph Prediction
Move into to `./ogbg` and run `train_graph.py` for graph prediction tasks.

## Transfer Learning
### Download Datasets
For ZINC-2M, fownload from [chem data](https://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put `./zinc_standard_agent` under `./dataset`.

For QM9, fownload from [MoleculeNet](https://moleculenet.org/datasets-1) and put the file `qm9.csv` under `./dataset/qm9/raw/qm9.csv`

### Running

(1) Move into `./qm9` and run `pretrain.py` to pre-train the encoder.

(2) Fine-tune the pre-trained encoder on QM9 by runing `tune_qm9.py`.


<!-- ## Cite -->
<!-- Welcome to kindly cite our work with:
```
@inproceedings{liugraph,
  title={Graph Distillation with Eigenbasis Matching},
  author={Liu, Yang and Bo, Deyu and Shi, Chuan},
  booktitle={Forty-first International Conference on Machine Learning}
}
``` -->
