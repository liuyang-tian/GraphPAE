import torch
from torch_scatter import scatter

from torch_geometric.nn import MessagePassing


class GatedGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        args,
    ):
        super(GatedGCNConv, self).__init__()

        self.dropout = args.gnn_dropout
        
        self.A1 = torch.nn.Linear(in_channels, out_channels)
        self.A2 = torch.nn.Linear(in_channels, out_channels)
        self.B1 = torch.nn.Linear(in_channels, out_channels)
        self.B2 = torch.nn.Linear(in_channels, out_channels)
        self.B3 = torch.nn.Linear(in_channels, out_channels)        

    def forward(self, h, p, edge_index, e, snorm_n):
        # For the e's
        B1_h = self.B1(h)
        B2_h = self.B2(h)
        B3_e = self.B3(e)

        # Step 1: Compute hat_eta and normalized eta
        row, col = edge_index
        hat_eta = B1_h[row] + B2_h[col] + B3_e

        eta_ij_bias = hat_eta
        eta_ij_bias = hat_eta + p
        pe = eta_ij_bias

        sigma_eta_ij_bias = torch.sigmoid(eta_ij_bias)
        sum_sigma_eta_ij_bias = scatter(sigma_eta_ij_bias, row, dim=0, reduce='sum', dim_size=h.shape[0])
        eta_ij = sigma_eta_ij_bias / (sum_sigma_eta_ij_bias[row] + 1e-6)

        # Step 2: Update h
        A1_h = self.A1(h)
        v_ij = self.A2(h[col])
        eta_mul_v = eta_ij * v_ij
        sum_eta_v = scatter(eta_mul_v, row, dim=0, reduce='sum', dim_size=h.shape[0])
        h = A1_h + sum_eta_v

        h = h * snorm_n

        return h, hat_eta, pe
