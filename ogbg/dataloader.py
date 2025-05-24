import torch
from torch_geometric.loader.dataloader import Collater

def collate_fn(graphs):
    collater = Collater(dataset=None)
    batch = collater(batch=graphs)
    tab_sizes_n = [graphs[i].num_nodes for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()
    batch.snorm_n = snorm_n
    
    return batch