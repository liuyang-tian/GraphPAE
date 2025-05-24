import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.utils.features import get_atom_feature_dims
from ogb.graphproppred.mol_encoder import BondEncoder

from utils import get_activation
from pos_enc import PEG
from conv import GatedGCNConv


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_atom_type):
        super(AtomEncoder, self).__init__()
        full_atom_feature_dims = get_atom_feature_dims()
        full_atom_feature_dims[0] = num_atom_type+1
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class GraphEncoder(torch.nn.Module):
    def __init__(self, out_dim, args=None):
        super().__init__()
        emb_dim = args.embed_dim
        self.x_embedding = AtomEncoder(emb_dim, args.num_atom_type)
        self.edge_embedding = BondEncoder(emb_dim)

        self.pe_enc = PEG(args=args)

        self.num_layer = args.enc_gnn_layer
        emb_dim = args.embed_dim
        self.gnn_dropout = args.gnn_dropout
        self.gnn_edge_dropout = args.gnn_edge_dropout
        
        self.gnns = nn.ModuleList()
        if self.num_layer == 1:
            self.gnns.append(GatedGCNConv(in_channels=args.embed_dim, out_channels=args.embed_dim, args=args))
        else:
            self.gnns.append(GatedGCNConv(in_channels=args.embed_dim, out_channels=args.embed_dim, args=args))
            self.activations = nn.ModuleList()
            self.activations_edge = nn.ModuleList()
            self.activations_pe = nn.ModuleList()
            for layer in range(self.num_layer - 1):
                self.gnns.append(GatedGCNConv(in_channels=args.embed_dim, out_channels=args.embed_dim, args=args))        
                self.activations.append(get_activation(args.gnn_activation))
                self.activations_edge.append(get_activation(args.gnn_activation))
                self.activations_pe.append(get_activation(args.gnn_activation))

        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_edge = torch.nn.ModuleList()
        self.batch_norms_pe = torch.nn.ModuleList()
        
        for layer in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_edge.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_pe.append(torch.nn.BatchNorm1d(emb_dim))


    def forward(self, x, x_masked, edge_index, edge_attr=None, snorm_n=None, PE=None, PE_noise=None):
        x = self.x_embedding(x)
        x_masked = self.x_embedding(x_masked)
        e = self.edge_embedding(edge_attr) if edge_attr is not None else None

        pe = self.pe_enc(PE)
        pe_noise = self.pe_enc(PE_noise)

        h_list = [x]
        e_h_list = [e]
        e_pe_list = [e]
        h_masked_list = [x_masked]
        pe_list = [pe]
        pe_noise_list = [pe_noise]
        for layer in range(self.num_layer):
            h_masked, e_h, pe = self.gnns[layer](h_masked_list[layer], pe_list[layer], edge_index, e_h_list[layer], snorm_n)

            h, e_pe, pe_noise = self.gnns[layer](h_list[layer], pe_noise_list[layer], edge_index, e_pe_list[layer], snorm_n)

            h_masked = self.batch_norms[layer](h_masked)
            e_h = self.batch_norms_edge[layer](e_h)
            h = self.batch_norms[layer](h)
            e_pe = self.batch_norms_edge[layer](e_pe)
            
            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
                e_pe = self.activations_edge[layer](e_pe)
                h_masked = self.activations[layer](h_masked)
                e_h = self.activations_edge[layer](e_h)
            
            h_masked = F.dropout(h_masked, p=self.gnn_dropout, training=self.training)
            e_h = F.dropout(e_h, p=self.gnn_edge_dropout, training=self.training)
            pe = F.dropout(pe, p=self.gnn_edge_dropout, training=self.training)

            h = F.dropout(h, p=self.gnn_dropout, training=self.training)
            e_pe = F.dropout(e_pe, p=self.gnn_edge_dropout, training=self.training)
            pe_noise = F.dropout(pe_noise, p=self.gnn_edge_dropout, training=self.training)
            
            h_masked_list.append(h_masked)
            h_list.append(h)
            e_h_list.append(e_h)
            e_pe_list.append(e_pe)
            pe_noise_list.append(pe_noise)
            pe_list.append(pe)
        
        return h_masked_list[-1], pe_noise_list[-1]

    def embed(self, x, edge_index, edge_attr=None, snorm_n=None, PE=None):
        x = self.x_embedding(x)
        e = self.edge_embedding(edge_attr) if edge_attr is not None else None
        pe = self.pe_enc(PE)

        h_list = [x]
        e_list = [e]
        pe_list = [pe]
        for layer in range(self.num_layer):
            h, e_h, pe = self.gnns[layer](h_list[layer], pe_list[layer], edge_index, e_list[layer], snorm_n)

            h = self.batch_norms[layer](h)
            e_h = self.batch_norms_edge[layer](e_h)
            
            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
                e_h = self.activations_edge[layer](e_h)
            
            h = F.dropout(h, p=self.gnn_dropout, training=self.training)
            e_h = F.dropout(e_h, p=self.gnn_edge_dropout, training=self.training)
            pe = F.dropout(pe, p=self.gnn_edge_dropout, training=self.training)
            
            h_list.append(h)
            e_list.append(e_h)
            pe_list.append(pe)
        
        return h_list[-1], pe_list[-1]