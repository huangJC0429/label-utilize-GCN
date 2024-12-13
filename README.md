# ELU-GCN:  Effectively Label-Utilizing Graph Convolutional Network

This repository contains an implementation of Enhancing the Influence of Labels on Unlabeled Nodes in Graph Convolutional Networks.

## Dependencies
- CUDA 10.2.89
- python 3.6.8
- pytorch 1.9.0
- pyg 2.0.3

## Usage
- For semi-supervised setting, run the following script
```sh
cd Citation
bash semi.sh
```

## Pretrain
- if you want to generate the file from the pretrain model, run the following command

```sh
python pre_train_MLP.py --dataset cora --hidden_dim 128
python pre_train_MLP.py --dataset citeseer --hidden_dim 256
python pre_train_MLP.py --dataset pubmed --hidden_dim 256
```
- For the Coauthor dataset
```sh
python pre_train_MLP_coauthor.py --dataset {dataset name}
```

