from itertools import repeat
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.io.read_graph_pyg import read_graph_pyg


class OGBDataset(PygGraphPropPredDataset):
    def __init__(self,
                 root,
                 dataset='hiv',
                 max_freqs=0,
                 lap_norm=None,
                 eigvec_norm="L2",
                 processed_name="processed",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):

        self.name = dataset
        self.processed_name = processed_name
        self.max_freqs = max_freqs
        self.lap_norm = lap_norm
        self.eigvec_norm = eigvec_norm
        self.root = root
        super(OGBDataset, self).__init__(name=dataset, transform=transform, pre_transform=pre_transform)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.Eig_list = torch.load(self.processed_paths[1])

    @property
    def processed_dir(self):
        return osp.join(self.root, self.processed_name)

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt', 'geometric_data_processed-eig.pt']


    def process(self) -> None:
        ### read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        Eig_list = [eigvec_precompute(data, self.lap_norm) for data in data_list]
        torch.save(Eig_list, self.processed_paths[1])

        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self._data.keys():
            if key!="num_nodes":
                item, slices = self._data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
                data[key] = item[s]
        # data.num_nodes = torch.IntTensor([data.x.shape[0]]).unsqueeze(-1)
        data.num_nodes = data.x.shape[0]

        EigVals, EigVecs = self.Eig_list[idx]
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=EigVals, evects=EigVecs,
            max_freqs=self.max_freqs,
            eigvec_norm=self.eigvec_norm,
        )

        return data
    

def eigvec_precompute(data, lap_norm):
    # Eigen Vectors Precomputing before pretraining
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.

    L_edge_index, L_values = get_laplacian(data.edge_index, normalization=lap_norm, num_nodes=N)
    L = to_scipy_sparse_matrix(L_edge_index, L_values)

    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals = torch.from_numpy(EigVals)
    EigVecs = torch.from_numpy(EigVecs)

    return EigVals, EigVecs


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2', skip_zero_freq=True, eigvec_abs=True):
    evals = evals.numpy()
    evects = evects.numpy()

    N = evects.shape[0]  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    offset = (abs(evals) < 1e-6).sum().clip(0, N) if skip_zero_freq else 0
    idx = evals.argsort()[offset:max_freqs + offset]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    # PE = torch.linalg.norm(evects[edge_index[0]] - evects[edge_index[1]], dim=-1)
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    
    if N < max_freqs + offset:
        EigVecs = F.pad(evects, (0, max_freqs + offset - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs + offset:
        EigVals = F.pad(evals, (0, max_freqs + offset - N),
                        value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)
    EigVecs = EigVecs.abs() if eigvec_abs else EigVecs

    return EigVals, EigVecs


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization in ["L1", "L2", "abs-max", "min-max"]:
        return normalizer(EigVecs, normalization, eps)

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def normalizer(x: torch.Tensor, normalization: str = "L2", eps: float = 1e-12):
    if normalization == "none":
        return x

    elif normalization == "L1":
        # L1 normalization: vec / sum(abs(vec))
        denom = x.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: vec / sqrt(sum(vec^2))
        denom = x.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: vec / max|vec|
        denom = torch.max(x.abs(), dim=0, keepdim=True).values

    elif normalization == "min-max":
        # MinMax normalization: (vec - min(vec)) / (max(vec) - min(vec))
        x = x - x.min(dim=0, keepdim=True).values
        denom = x.max(dim=0, keepdim=True).values

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    return x / denom.clamp_min(eps).expand_as(x)

