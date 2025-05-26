import os
import argparse
import math
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from load_dataset import MoleculeDataset
from dataloader import collate_fn
from utils import load_config, create_optimizer

from autoencoder import GraphAutoEncoder
from encoder import GraphEncoder


def train_mae_epoch(accelerator, graph_auto_encoder, data_loader, optimizer, epoch):
    graph_auto_encoder.train()
    sum_loss, count = 0.0, 0
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process) as pbar:
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            loss = graph_auto_encoder(batch)

            accelerator.backward(loss)
            optimizer.step()
            
            total_loss = accelerator.gather(loss.detach())
            total_graph = accelerator.gather(torch.tensor(batch.num_graphs, device=accelerator.device))
            sum_loss += (total_loss * total_graph).sum()            
            count += total_graph.sum()
            avg_loss = sum_loss / count

            if (step+1) % 50 == 0:
                pbar.set_postfix({'AvgLoss': avg_loss.item()})
                pbar.update(50)

    accelerator.print(f"Num of graphs: {count}, AvgLoss: {avg_loss.item()}")
    return avg_loss


def main(args):
    log_dir = './{}'.format(args.dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    set_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    accelerator = Accelerator(cpu=False)
    accelerator.print(args)

    processed_name = f"processed_{args.lap_norm}"
    dataset = MoleculeDataset(root=args.root, dataset=args.dataset, max_freqs=args.max_freqs, lap_norm=args.lap_norm, processed_name=processed_name)

    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = GraphEncoder(out_dim=args.num_atom_type, args=args)
    model = GraphAutoEncoder(encoder=encoder, args=args)
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

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    for epoch in range(1 + args.resume_epochs, args.resume_epochs + args.epochs + 1):
        train_mae_epoch(accelerator=accelerator, graph_auto_encoder=model, data_loader=train_loader, optimizer=optimizer, epoch=epoch)
        if scheduler:
            scheduler.step()

        if epoch % args.save_epochs == 0:
            accelerator.wait_for_everyone()
            enc_model = accelerator.get_state_dict(encoder)
            autoencoder_model = accelerator.get_state_dict(model)
            torch.save(enc_model, os.path.join(log_dir, f'encoder_{epoch}.pth'))
            torch.save(autoencoder_model, os.path.join(log_dir, f'autoencoder_model_{epoch}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--root', type=str, default="../dataset")
    parser.add_argument('--dir_name', type=str, default="./mask_atom_noise_pe")
    parser.add_argument("--dataset", type=str, default="zinc_tmp")  # zinc_standard_agent
    args = parser.parse_args()
    
    config = load_config(f"./config/pretraining_on_zinc.yaml")
    for key, value in config.items():
        setattr(args, key, value)

    main(args)

