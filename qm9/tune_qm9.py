import os
import copy
from argparse import ArgumentParser

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import load_dataset
from load_dataset import MoleculeDataset
from dataloader import collate_fn

from splitters import scaffold_split

from encoder import GraphEncoder
from model import Model

from utils import AverageMeter, set_random_seed, load_config, create_optimizer


def train_epoch(model, optimizer, data_loader, device):
    criterion = nn.L1Loss()
    model.train()
    optimizer.zero_grad()
    loss_meter = AverageMeter()
    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        pred = model(batch).to(torch.float64)
        y = batch.y.view(pred.shape).to(torch.float64)

        pred = pred * label_std + label_mean
        loss = criterion(pred, y)  # shape = [N, C]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(float(loss), pred.shape[0])

    return loss_meter.avg


def evaluate_network(model, data_loader, device):
    model.eval()
    y_true = []
    y_scores = []

    for batch in data_loader:
        batch = copy.copy(batch)
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        pred = pred * label_std + label_mean
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    mae_scores = np.mean(np.abs(y_scores - y_true), axis=0)
    mean_mae_score = float(np.mean(mae_scores))
    return mean_mae_score, mae_scores


parser = ArgumentParser()
parser.add_argument('--num_exp', type=int, default=5)
parser.add_argument('--root', type=str, default="../dataset")
parser.add_argument('--dir_name', type=str, default="./mask_atom_noise_pe")
parser.add_argument('--dataset', type=str, default="qm9")
parser.add_argument('--split', type=str, default="scaffold")
parser.add_argument('--norm_label', type=bool, default=False)
parser.add_argument('--pe_type', type=str, default="peg")

args = parser.parse_args()

config = load_config(f"./config/tune_on_qm9.yaml")
for key, value in config.items():
    setattr(args, key, value)

args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
print(args)

accs = []
for ep_num in range(args.num_exp):
    set_random_seed(ep_num)
    torch.cuda.manual_seed_all(ep_num)

    processed_name = f"processed_{args.lap_norm}"
    dataset = MoleculeDataset(root=args.root, dataset=args.dataset, max_freqs=args.max_freqs, lap_norm=args.lap_norm, processed_name=processed_name)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('../dataset/' + args.dataset + f'/{processed_name}/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)

    if args.norm_label:
        raw_name = os.listdir(f'../dataset/{args.dataset}/raw/')[0]
        load_labels = getattr(load_dataset, f'_load_{args.dataset}_dataset')
        _, _, labels = np.array(load_labels(f'dataset/{args.dataset}/raw/{raw_name}'))
        label_mean, label_std = np.mean(labels), np.std(labels)
        print(f'label_mean of {args.dataset}: {label_mean}')
        print(f'label_std of {args.dataset}: {label_std}')
    else:
        label_mean, label_std = 0, 1

    encoder = GraphEncoder(out_dim=args.embed_dim, args=args)
    encoder.load_state_dict(torch.load(os.path.join(args.dir_name, f'encoder_100.pth'), map_location="cpu"))
    model = Model(encoder=encoder, args=args).to(args.device)
    parameters = model.parameters()

    if args.optim == "sgd":
        pass
    else:
        args.momentum = None        
    optimizer = create_optimizer(opt=args.optim, parameters=parameters, lr=args.init_lr, weight_decay=float(args.weight_decay), momentum=args.momentum)
    
    if args.use_scheduler:
        def lr_lambda_cosine(current_step):
            num_cycles = 0.5
            if current_step < args.num_warmup_steps:
                return max(1e-6, float(current_step) / float(max(1, args.num_warmup_steps)))
            progress = float(current_step - args.num_warmup_steps) / float(max(1, args.epochs - args.num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda_cosine, -1)
    else:
        scheduler = None

    train_list = []
    train_tasks_list = []
    val_list = []
    val_tasks_list = []
    test_list = []
    test_tasks_list = []
    best_idx = 0
    for epoch in range(1, args.epochs+1):
        train_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=args.device)
        if scheduler:
            scheduler.step()

        train_score, train_tasks_scores = evaluate_network(model=model, data_loader=train_loader, device=args.device)
        val_score, val_tasks_scores = evaluate_network(model=model, data_loader=val_loader, device=args.device)
        test_score, test_tasks_scores = evaluate_network(model=model, data_loader=test_loader, device=args.device)

        train_list.append(train_score)
        train_tasks_list.append(train_tasks_scores)
        val_list.append(val_score)
        val_tasks_list.append(val_tasks_scores)
        test_list.append(test_score)
        test_tasks_list.append(test_tasks_scores)

        if val_score < val_list[best_idx]:
            best_idx = epoch - 1

        print(f"Epoch {epoch}, train_mae: {train_score}, val_mae: {val_score}, test_mae: {test_score}")
        print(f"Test_mae_scores: {list(test_tasks_scores)}")


    print(f"Test_tasks_scores: {list(test_tasks_scores)}")
    print(f"Exp {ep_num}, train_score: {train_score}, val_score: {val_score}, test_score: {test_score}")
    accs.append(test_score)

mean_acc = np.mean(accs)
std_acc = np.std(accs)
print(f"final results: mean acc: {mean_acc}\t std: {std_acc}")
