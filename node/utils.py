import os
import yaml
import random
import math
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Adagrad, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()

            
def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split(y):
    train_ratio = 0.1
    val_ratio = 0.1
    test_ratio = 0.8

    N = len(y)
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_idx)

    return train_idx, val_idx, test_idx 


def create_optimizer(opt, parameters, lr, weight_decay, momentum):
    if opt == "adam":
        optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt == "adamw":
        optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        optimizer = SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif opt == "adagrad":
        optimizer = Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif opt == "rmsprop":
        optimizer = RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def get_activation(activation):
    if activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    elif activation == 'tanh':
        activation_fn = nn.Tanh()
    elif activation == 'prelu':
        activation_fn = nn.PReLU()
    elif activation == 'gelu':
        activation_fn = nn.GELU()
    else:
        raise ValueError('Invalid activation')
    return activation_fn


def compute_loss(pred, true, loss_fun, task_type, size_average):
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction=size_average)
    mse_loss = torch.nn.MSELoss(reduction=size_average)

    if loss_fun == 'cross_entropy':
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true)
        else:
            if task_type=='classification_multilabel':
                is_labeled = true == true
                return bce_loss(pred[is_labeled], true[is_labeled].float())
            else:
                return bce_loss(pred, true.float())
        
    elif loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true)
    else:
        raise ValueError(f"Loss function '{loss_fun}' not supported")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert n > 0
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def noise_fn(noise, num_mask, dim):
    noise_array = np.random.uniform(low=-noise, high=noise, size=(num_mask, dim))
    return torch.tensor(noise_array, dtype=torch.float32)


def cal_relative_mat(pe_embed, batch, norm):
    pad_pe_embed, _ = to_dense_batch(pe_embed, batch)  # [b, n_max, pe_dim], [b, n_max]
    if norm:
        pad_pe_embed = F.normalize(pad_pe_embed, p=2, dim=1)  # [b, n_max, pe_dim]
    pad_pe_embed = pad_pe_embed.permute(0, 2, 1)   # [b, pe_dim, n_max]

    pad_pe_embed = pad_pe_embed.unsqueeze(-1)  # [b, pe_dim, n_max] -> [b, pe_dim, n_max, 1]
    pad_pe_embed_t = pad_pe_embed.permute(0, 1, 3, 2)  # [b, pe_dim, n_max, 1] -> [b, pe_dim, 1, n_max]
    pe_mat = torch.matmul(pad_pe_embed, pad_pe_embed_t)  # [b, pe_dim, n_max, 1] @ [b, pe_dim, 1, n_max] -> [b, pe_dim, n_max, n_max]

    return pe_mat
