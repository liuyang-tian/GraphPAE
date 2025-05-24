from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops, to_undirected
from utils import get_activation, noise_fn


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

    def forward(self, dist, edge_index_pe):
        edge_index, dist = remove_self_loops(edge_index=edge_index_pe, edge_attr=dist)

        dist = self.dense(dist)
        dist = self.activation_fn(dist)
        dist = self.layer_norm(dist)
        dist = self.out_proj(dist)  # [e_sum, head] -> [n_sum, 1]
        
        edge_index, dist = to_undirected(edge_index=edge_index, edge_attr=dist, reduce="mean")        
        return dist.squeeze(), edge_index


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
        

class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, num_atom_type=0, args=None):
        super(GraphAutoEncoder, self).__init__()
        self.args = args
        self.encoder = encoder

        self.mask_ratio = args.mask_ratio
        self.replace_ratio = args.replace_ratio
        self.noise_val = args.noise_val
        self.masked_atom_loss = float(args.masked_atom_loss)
        self.masked_pe_loss = float(args.masked_pe_loss)
        self.atom_recon_type = args.atom_recon_type
        self.num_atom_type = num_atom_type
        self.alpha_l = args.alpha_l

        self.enc_mask_token = nn.Parameter(torch.zeros(1, args.feat_dim))
        self.node_pred = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)
        self.pe_reconstruct_heads = DistanceHead(heads=args.heads, activation_fn=args.task_head_activation)


    def forward(self, x, edge_index, u, PE, edge_index_pe=None):
        x_masked, u_masked, mask_tokens = self.encoding_mask_noise(x=x, u=u, mask_ratio=self.mask_ratio, replace_ratio=self.replace_ratio)
        
        PE_noise = torch.linalg.norm(u_masked[edge_index_pe[0]] - u_masked[edge_index_pe[1]], dim=-1)
        enc_rep, pe = self.encoder(x, x_masked, edge_index, PE=PE, PE_noise=PE_noise)
        
        enc_rep = self.node_pred(enc_rep, mask_tokens)
        pe_loss = 0.0
        reconstruct_dist, _ = self.pe_reconstruct_heads(pe, edge_index_pe)
        atom_loss = self.cal_atom_loss(pred_node=enc_rep, target_atom=x, mask_tokens=mask_tokens,
                                loss_fn=self.atom_recon_type, alpha_l=self.alpha_l)
        pe_loss = self.cal_pe_loss(reconstruct_dis=reconstruct_dist, target_dis=PE, edge_index_pe=edge_index_pe, mask_tokens=mask_tokens)
        loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss

        return loss
        
    def encoding_mask_noise(self, x, u, mask_ratio, replace_ratio):
        mask_token_ratio = 1 - replace_ratio
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        if replace_ratio > 0:
            num_noise_nodes = int(replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(mask_token_ratio * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(replace_ratio * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        u_masked = None

        u_masked = u.clone()
        pos_noise = noise_fn(self.noise_val, len(mask_nodes), u.size(1)).to(u_masked.device)
        u_masked[mask_nodes] += pos_noise

        return out_x, u_masked, mask_nodes
    

    def embed(self, x, edge_index, PE):        
        enc_rep, _ = self.encoder.embed(x, edge_index, PE=PE)        
        return enc_rep


    def cal_pe_loss(self, reconstruct_dis, target_dis, edge_index_pe, mask_tokens):
        edge_index, target_dis = remove_self_loops(edge_index=edge_index_pe, edge_attr=target_dis)
        edge_index, target_dis = to_undirected(edge_index=edge_index, edge_attr=target_dis, reduce="mean")
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
        target_atom = target_atom[mask_tokens]

        if loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
            atom_loss = criterion(pred_node, target_atom)
        elif loss_fn == "mse":
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