import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.typing import SparseTensor, torch_sparse
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        args,
        concat = True,
        negative_slope = 0.2,
        add_self_loops = True,
        edge_dim = None,
        fill_value = 'mean',
        bias = True,
        residual = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dropout = args.gnn_edge_dropout
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = nn.Linear(in_channels, self.heads * out_channels, bias=False)
        else:
            self.lin_src = nn.Linear(in_channels[0], self.heads * out_channels, False)
            self.lin_dst = nn.Linear(in_channels[1], self.heads * out_channels, False)

        self.att_src = Parameter(torch.FloatTensor(size=(1, self.heads, out_channels)))
        self.att_dst = Parameter(torch.FloatTensor(size=(1, self.heads, out_channels)))

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, self.heads * out_channels, bias=False)
            self.att_edge = Parameter(torch.FloatTensor(size=(1, self.heads, out_channels)))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        total_out_channels = out_channels * (self.heads if concat else 1)

        if residual:
            self.res = nn.Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.FloatTensor((total_out_channels)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'lin'):
            nn.init.xavier_normal_(self.lin.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.lin_src.weight, gain=gain)
            nn.init.xavier_normal_(self.lin_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.att_src, gain=gain)
        nn.init.xavier_normal_(self.att_dst, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res, nn.Linear):
            nn.init.xavier_normal_(self.res, gain=gain)

    def forward(
        self,
        x,
        pe,
        edge_index,
        edge_attr = None,
        size = None,
        return_attention_weights = None,
    ):

        H, C = self.heads, self.out_channels
        res = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                raw_edge_index = edge_index
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    raw_edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
                # if pe is not None:
                #     edge_index_pe, pe = remove_self_loops(raw_edge_index, pe)
                #     _, pe = add_self_loops(edge_index_pe, pe, fill_value=1, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha, pe_out = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
                                  size=size, pe=pe)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias
        
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, pe_out

    def edge_update(self, alpha_j, alpha_i,
                    edge_attr, index, ptr,
                    dim_size, pe):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = alpha + pe
        pe_out = alpha
            
        alpha = F.leaky_relu(alpha, self.negative_slope)

        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.edge_dropout, training=self.training)

        return alpha, pe_out

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels},'
                f'{self.out_channels}, heads={self.heads})')