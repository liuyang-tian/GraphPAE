import argparse

import numpy as np
import torch
from torch import optim as optim
from torch.utils.data import DataLoader

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from ogb.graphproppred import Evaluator

from dataset_ogb import OGBDataset
from dataloader import collate_fn

from autoencoder import GraphAutoEncoder
from encoder import GraphEncoder
from evaluation import ogbg_evaluation
from utils import load_config, create_optimizer, create_schedule, set_random_seed


def train_mae_epoch(graph_auto_encoder, data_loader, optimizer, device):
    graph_auto_encoder.train()
    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = graph_auto_encoder(batch)
        loss.backward()
        optimizer.step()
    

def evaluation_network(model, data_loader, pooling, device):
    model.eval()
    emb = []
    y_true = []

    for batch in data_loader:
        with torch.no_grad():
            batch = batch.to(device)
            out = model.embed(batch)
            if pooling == "mean":
                out = global_mean_pool(out, batch.batch)
            elif pooling == "max":
                out = global_max_pool(out, batch.batch)
            elif pooling == "sum":
                out = global_add_pool(out, batch.batch)
            else:
                raise NotImplementedError
            
        emb.append(out.detach().cpu())
        y_true.append(batch.y.cpu())

    emb = torch.cat(emb, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()    
    return emb, y_true


def main(args):
    final_results = []
    for ep_num in range(args.num_exp):
        args.seed = ep_num
        set_random_seed(ep_num)
        torch.cuda.manual_seed_all(ep_num)

        processed_name = f"processed_{args.lap_norm}"
        dataset = OGBDataset(root=args.root, dataset=args.dataset, max_freqs=args.max_freqs, lap_norm=args.lap_norm, processed_name=processed_name)
        
        split_idx = dataset.get_idx_split()
        task_type = dataset.task_type
        args.num_tasks = dataset.num_tasks

        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
        evaluator = Evaluator(name=args.dataset)

        encoder = GraphEncoder(out_dim=args.num_atom_type, args=args).to(args.device)
        model = GraphAutoEncoder(encoder=encoder, num_atom_type=args.num_atom_type, args=args).to(args.device)
        parameters = model.parameters()

        if args.optim == "sgd":
            momentum = args.momentum
        else:
            momentum = None
        optimizer = create_optimizer(opt=args.optim, parameters=parameters, lr=args.init_lr, weight_decay=float(args.weight_decay), momentum=momentum)
        if args.use_scheduler:
            scheduler = create_schedule(opt=args.schedule_opt, optimizer=optimizer, max_epoch=args.epochs, num_warmup_steps=args.num_warmup_epochs)
        else:
            scheduler = None

        for epoch in range(1, args.epochs+1):
            train_mae_epoch(graph_auto_encoder=model, data_loader=train_loader, optimizer=optimizer, device=args.device)
            if scheduler:
                scheduler.step()

        embed, y_true = evaluation_network(model, valid_loader, args.pooling, args.device)
        train_score, valid_score, test_score = ogbg_evaluation(embed, y_true, split_idx, evaluator, task_type, args.num_tasks)
        
        final_results.append(test_score)     
               
    print(f"{final_results}")
    mean_final_result = np.mean(final_results)
    std_final_result = np.std(final_results)
    print(f"final result: {mean_final_result:.5f}±{std_final_result:.5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=5)
    parser.add_argument('--root', type=str, default="./dataset")
    parser.add_argument("--dataset", type=str, default="ogbg-molbace")
    args = parser.parse_args()

    config = load_config(f"./config/{args.dataset}.yaml")
    for key, value in config.items():
        setattr(args, key, value)

    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.schedule_opt == "none":
        args.use_scheduler = False
    else:
        args.use_scheduler = True
    
    print(args)
    main(args)

