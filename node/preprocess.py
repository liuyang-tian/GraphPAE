import os
import random
import torch
import torch.nn as nn
import numpy as np
import scipy
import scipy.stats as st

from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected, degree
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.utils import get_laplacian
from torch_geometric.transforms import ToUndirected
import scipy as sp
import scipy.sparse as sps
import time
from scipy.io import loadmat
from collections import Counter
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Actor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def load_blog(path="../dataset/"):
    dataset = "blog"
    # print('Loading {} dataset...'.format(dataset))
    adj = sps.load_npz(path+dataset+"/adj.npz")
    features = sps.load_npz(path+dataset+"/feat.npz")
    labels = np.load(path+dataset+"/label.npy")
    idx_train20 = np.load(path+dataset+"/train20.npy")
    idx_val = np.load(path+dataset+"/val.npy")
    idx_test = np.load(path+dataset+"/test.npy")

    adj = adj.todense()
    row, col = np.where(adj != 0)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    x = torch.tensor(features.todense(), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.int64)
    data = Data(x=x, edge_index=edge_index, y=y)

    num_nodes = x.size(0)  # Get the number of nodes in the dataset
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[idx_train20] = 1
    val_mask[idx_val] = 1  # Set the validation indices to 1
    test_mask[idx_test] = 1  # Set the test indices to 1
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


## load blog
data = load_blog()


### load chameleon
# dataset = WikipediaNetwork(root='../dataset/WikipediaNetwork', name='chameleon', transform=NormalizeFeatures())
# data = dataset[0]


### load squirrel
# dataset = WikipediaNetwork(root='../dataset/WikipediaNetwork', name='squirrel', transform=NormalizeFeatures())
# data = dataset[0]


### load actor
# dataset = Actor(root='../dataset/Actor', transform=NormalizeFeatures())
# data = dataset[0]


if data.is_directed():
    print("is not undirected")
    data.edge_index = to_undirected(data.edge_index)

index, attr = get_laplacian(data.edge_index, normalization='sym')
L = to_scipy_sparse_matrix(index, attr)

L = torch.FloatTensor(L.todense())
e, u = torch.linalg.eigh(L)

# e, u = scipy.sparse.linalg.eigsh(L, k=800, which='SA', tol=1e-5)

data.e = torch.FloatTensor(e)
data.u = torch.FloatTensor(u)

torch.save(data, '../dataset/blog.pt')