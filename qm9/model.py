import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class Model(nn.Module):
    def __init__(self, encoder, args=None):
        super(Model, self).__init__()
        self.encoder = encoder

        if args.pooling == "mean":
            self.pool = global_mean_pool
        elif args.pooling == "max":
            self.pool = global_max_pool
        elif args.pooling == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        self.output_layer = nn.Linear(args.embed_dim, args.num_tasks)

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        snorm_n = batch.snorm_n

        u = None
        PE, edge_index_pe = None, None
        e, u = batch.EigVals.clone(), batch.EigVecs.clone()
        mask_u = torch.isnan(u)
        u[mask_u] = 0   # [n_sum, max_freqs]
        mask_e = torch.isnan(e)
        e[mask_e] = 0

        PE = torch.linalg.norm(u[edge_index[0]] - u[edge_index[1]], dim=-1)

        enc_rep, pe = self.encoder.embed(x, edge_index, edge_attr=edge_attr, snorm_n=snorm_n, PE=PE)

        graph_rep = self.pool(enc_rep, batch.batch)  # [b, d]
        return self.output_layer(graph_rep)  # [b, num_task]