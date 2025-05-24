from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_undirected

from utils import noise_fn, get_activation


class MaskLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, mask_tokens=None):
        if mask_tokens is not None:
            features = features[mask_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias

        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.out_proj = nn.Linear(128, 1)
        self.activation_fn = get_activation(activation_fn)

    def forward(self, dist, edge_index):
        dist = self.dense(dist)
        dist = self.activation_fn(dist)
        dist = self.layer_norm(dist)
        dist = self.out_proj(dist)
        
        edge_index, dist = to_undirected(edge_index=edge_index, edge_attr=dist, reduce="mean")
        return dist.squeeze(), edge_index


class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, num_atom_type=0, args=None):
        super(GraphAutoEncoder, self).__init__()
        self.args = args
        self.encoder = encoder

        self.mask_ratio = args.mask_ratio
        self.noise_val = args.noise_val

        self.masked_atom_loss = float(args.masked_atom_loss)
        self.masked_pe_loss = float(args.masked_pe_loss)
        self.atom_recon_type = args.atom_recon_type
        self.num_atom_type = args.num_atom_type
        self.alpha_l = args.alpha_l

        self.node_pred = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)
        self.pe_reconstruct_heads = DistanceHead(heads=args.embed_dim, activation_fn=args.task_head_activation)

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        snorm_n = batch.snorm_n

        u = None
        PE = None, None
        e, u = batch.EigVals.clone(), batch.EigVecs.clone()
        mask_u = torch.isnan(u)
        u[mask_u] = 0   # [n_sum, max_freqs]
        mask_e = torch.isnan(e)
        e[mask_e] = 0
        PE = torch.linalg.norm(u[edge_index[0]] - u[edge_index[1]], dim=-1)

        x_masked, u_masked, mask_tokens = self.encoding_mask_noise(x=x, u=u, mask_ratio=self.mask_ratio)
        
        PE_noise = torch.linalg.norm(u_masked[edge_index[0]] - u_masked[edge_index[1]], dim=-1)
        enc_rep, pe = self.encoder(x, x_masked, edge_index, edge_attr=edge_attr, snorm_n=snorm_n, PE=PE, PE_noise=PE_noise)
        
        enc_rep = self.node_pred(enc_rep, mask_tokens)
        pe_loss = 0.0
            
        reconstruct_dist, _ = self.pe_reconstruct_heads(pe, edge_index)
        atom_loss = self.cal_atom_loss(pred_node=enc_rep, target_atom=x, mask_tokens=mask_tokens,
                                loss_fn=self.atom_recon_type, alpha_l=self.alpha_l)
        pe_loss = self.cal_pe_loss(reconstruct_dis=reconstruct_dist, target_dis=PE, edge_index=edge_index, mask_tokens=mask_tokens)
        loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss
        return loss
        
    def encoding_mask_noise(self, x, u, mask_ratio):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        out_x = x.clone()
        out_x[mask_nodes, 0] = self.num_atom_type

        u_masked = None
        u_masked = u.clone()
        pos_noise = noise_fn(self.noise_val, len(mask_nodes), u.size(1)).to(u_masked.device)
        u_masked[mask_nodes] += pos_noise

        return out_x, u_masked, mask_nodes
    

    def embed(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        snorm_n = batch.snorm_n
        
        u = None
        PE = None
        e, u = batch.EigVals.clone(), batch.EigVecs.clone()
        mask_u = torch.isnan(u)
        u[mask_u] = 0   # [n_sum, max_freqs]
        mask_e = torch.isnan(e)
        e[mask_e] = 0

        PE = torch.linalg.norm(u[edge_index[0]] - u[edge_index[1]], dim=-1)

        enc_rep, _ = self.encoder.embed(x, edge_index, edge_attr=edge_attr, snorm_n=snorm_n, PE=PE)        
        return enc_rep


    def cal_pe_loss(self, reconstruct_dis, target_dis, edge_index, mask_tokens):
        edge_index, target_dis = to_undirected(edge_index, target_dis, reduce="mean")
        row = edge_index[0]
        idx = torch.isin(row, mask_tokens)

        reconstruct_dis = reconstruct_dis[idx]
        target_dis = target_dis[idx]

        pe_reconstruct_loss = F.smooth_l1_loss(
            reconstruct_dis,
            target_dis,
            reduction="mean",
            beta=1.0,
        )

        return pe_reconstruct_loss

    def cal_atom_loss(self, pred_node, target_atom, mask_tokens, loss_fn, alpha_l=0.0):
        target_atom = target_atom[mask_tokens, 0]

        if loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
            target_atom = F.one_hot(target_atom, num_classes=self.num_atom_type).float()
            atom_loss = criterion(pred_node, target_atom)
        elif loss_fn == "mse":
            target_atom = F.one_hot(target_atom, num_classes=self.num_atom_type).float()
            atom_loss = self.mse_loss(pred_node, target_atom)
        else:
            criterion = nn.CrossEntropyLoss()
            atom_loss = criterion(pred_node, target_atom)

        return atom_loss

    def sce_loss(self, x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
        return loss