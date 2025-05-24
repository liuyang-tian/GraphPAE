from os import path
import numpy as np

import scipy
from sklearn.preprocessing import label_binarize

import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected


DATAPATH = '../dataset/'

def load_fb100(filename):
    mat = scipy.io.loadmat(DATAPATH + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int64)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]

    data = Data()
    data.x = node_feat
    data.edge_index = edge_index
    data.num_nodes = num_nodes
    data.y = torch.tensor(label)

    return data


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


data = load_fb100_dataset('Penn94')

splits = np.load(f'../dataset/fb100-Penn94-splits.npy', allow_pickle=True)
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

torch.save(data, f'../dataset/penn94.pt')
