from os import path
import numpy as np

import scipy
import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
from torch_geometric.transforms import ToUndirected

from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset


def even_quantile_labels(vals, nclasses, verbose=True):
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def split_to_mask(data, splits_lst):
    train_mask_list = []
    val_mask_list = []
    test_mask_list = []
    for i in range(len(splits_lst)):
        split = splits_lst[i]
        train_mask = torch.LongTensor([0]*data.num_nodes)
        train_mask[split['train']] = 1
        train_mask = train_mask.bool()
        
        val_mask = torch.LongTensor([0]*data.num_nodes)
        val_mask[split['valid']] = 1
        val_mask = val_mask.bool()
        
        test_mask = torch.LongTensor([0]*data.num_nodes)
        test_mask[split['test']] = 1
        test_mask = test_mask.bool()

        train_mask_list.append(train_mask.unsqueeze(-1))
        val_mask_list.append(val_mask.unsqueeze(-1))
        test_mask_list.append(test_mask.unsqueeze(-1))
    
    train_mask = torch.cat(train_mask_list, dim=-1)
    val_mask = torch.cat(val_mask_list, dim=-1)
    test_mask = torch.cat(test_mask_list, dim=-1)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    data.y = data.y.view(-1)
    
    return data


nclass=5
dataset = PygNodePropPredDataset('ogbn-arxiv', root='../dataset', transform=ToUndirected())

label = even_quantile_labels(dataset.node_year.flatten(), nclass, verbose=False)
data = dataset[0]
data.y = torch.as_tensor(label)

splits = np.load(f'../dataset/arxiv-year-splits.npy', allow_pickle=True)
for i in range(len(splits)):
    for key in splits[i]:
        if not torch.is_tensor(splits[i][key]):
            splits[i][key] = torch.as_tensor(splits[i][key])

data = split_to_mask(data, splits)

if data.is_directed():
    print("is not undirected")
    data.edge_index = to_undirected(data.edge_index)

index, attr = get_laplacian(data.edge_index, normalization='sym')
L = to_scipy_sparse_matrix(index, attr)

e, u = scipy.sparse.linalg.eigsh(L, k=1000, which='SA', tol=1e-5)

data.e = torch.FloatTensor(e)
data.u = torch.FloatTensor(u)

torch.save(data, '../dataset/arxiv-year.pt')
