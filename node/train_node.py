
from argparse import ArgumentParser
import numpy as np

import torch

from utils import set_random_seed, split, load_config, create_optimizer
from evaluation import node_evaluation

from encoder import GraphEncoder
from autoencoder import GraphAutoEncoder
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)


def train_mae_epoch(graph_auto_encoder, x, edge_index, edge_index_pe, u, PE, optimizer):
    graph_auto_encoder.train()
    loss = graph_auto_encoder(x, edge_index, u, PE, edge_index_pe)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


parser = ArgumentParser()
parser.add_argument('--num_exp', type=int, default=5)
parser.add_argument('--root', type=str, default="./dataset")
parser.add_argument('--dataset', type=str, default="blog")  # ["blog", "chameleon", "squirrel", "actor"]
args = parser.parse_args()

config = load_config(f"./config/{args.dataset}.yaml")
for key, value in config.items():
    setattr(args, key, value)

args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

print(args)

data = torch.load('../dataset/{}.pt'.format(args.dataset))
x = data.x.float().to(args.device)
edge = data.edge_index.long().to(args.device)
e = data.e[:args.max_freqs].float().to(args.device)
u = data.u[:, :args.max_freqs].float().to(args.device)

y = data.y.to(args.device)
print(y.min().item(), y.max().item())
nclass = y.max().item() + 1

edge_index_pe, _ = remove_self_loops(edge, None)
edge_index_pe, _ = add_self_loops(edge_index_pe, fill_value='mean', num_nodes=u.shape[0])
PE = torch.linalg.norm(u[edge_index_pe[0]] - u[edge_index_pe[1]], dim=-1)  # [e_sum, 1]

final_results = []
for ep_num in range(args.num_exp):
    args.seed = ep_num
    set_random_seed(ep_num)

    if 'train_mask' in data.keys():
        if len(data.train_mask.size()) > 1:
            train_idx = torch.where(data.train_mask[:, args.seed])[0]
            val_idx = torch.where(data.val_mask[:, args.seed])[0]
            test_idx = torch.where(data.test_mask)[0]
        else:
            train_idx = torch.where(data.train_mask)[0]
            val_idx = torch.where(data.val_mask)[0]
            test_idx = torch.where(data.test_mask)[0]
    else:
        train_idx, val_idx, test_idx = split(y)

    encoder = GraphEncoder(out_dim=args.embed_dim, args=args).to(args.device)
    model = GraphAutoEncoder(encoder=encoder, num_atom_type=args.feat_dim, args=args).to(args.device)

    parameters = model.parameters()

    if args.optim == "sgd":
        pass
    else:
        args.momentum = None
    optimizer = create_optimizer(opt=args.optim, parameters=parameters, lr=args.init_lr, weight_decay=float(args.weight_decay), momentum=args.momentum)

    if args.use_schedule:
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    for epoch in range(1, args.epochs+1):
        train_mae_epoch(graph_auto_encoder=model, x=x, edge_index=edge, u=u, PE=PE, edge_index_pe=edge_index_pe, optimizer=optimizer)
        if scheduler:
            scheduler.step()
            
    model.eval()
    with torch.no_grad():
        embed = model.embed(x, edge, PE)
    acc, pred = node_evaluation(emb=embed, y=y, train_idx=train_idx, valid_idx=val_idx, test_idx=test_idx, epochs=args.epochs_eval, lr=args.lr_eval, weight_decay=args.wd_eval)
    print(f"Epoch {epoch}, ACC: {acc.item()}")

    final_results.append(acc.item())

mean_final_result = np.mean(final_results)
std_final_result = np.std(final_results)
print(f"{final_results}")
print(f"final result: {mean_final_result:.5f}±{std_final_result:.5}")
