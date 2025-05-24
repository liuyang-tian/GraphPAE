import torch
import torch.nn as nn

from utils import get_activation
from torch_geometric.nn import Linear


class LapPE(nn.Module):
    def __init__(self, pe_dim, pos_activation):
        super().__init__()
        self.linear_A = Linear(2, 2 * pe_dim)
        self.pe_encoder = nn.Sequential(get_activation(pos_activation), Linear(2 * pe_dim, pe_dim), get_activation(pos_activation))

    def forward(self, eigvals, eigvecs):
        if self.training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((eigvecs.unsqueeze(2), eigvals), dim=2) # [n, k, 2]
        mask_eig = torch.isnan(pos_enc)
        pos_enc[mask_eig] = 0

        pos_enc = self.linear_A(pos_enc) # [n, k, d]
        pos_enc = self.pe_encoder(pos_enc)
        pos_enc = pos_enc.clone().masked_fill_(mask_eig[:, :, 0].unsqueeze(2), 0.)
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # [n, d]

        return pos_enc


class PEG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gbf = nn.Sequential(GaussianLayer(K=128))
        self.gbf_proj = NonLinearHead(input_dim=128, out_dim=args.embed_dim, activation_fn=args.pos_activation)
        # self.gbf = nn.Sequential(GaussianLayer(K=args.heads), nn.Sigmoid())
        # self.gbf = nn.Sequential(nn.Linear(1, 32), nn.Linear(32, args.heads), nn.Sigmoid())

    def forward(self, PE):
        PE = PE.view([-1, 1])  # [e_sum, 1]
        pos_enc = self.gbf(PE)  # [e_sum, d]
        pos_enc = self.gbf_proj(pos_enc)
        return pos_enc  # [e_sum, heads]



class NonLinearHead(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
    

class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 0, 3)  # 0.1
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x):
        x = x.expand(-1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
