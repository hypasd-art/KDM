"""
The code is copied and adapted from https://arxiv.org/abs/2209.14734
"""
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
import sys
sys.path.append("..")

from datasets.abstract_dataset import AbstractDatasetInfos



class DataModule(nn.Module):
    def __init__(self, args):
        self.args = args
        self.dataloaders = None
        self.input_dims = None
        self.output_dims = None
        self.datadir = args.dataset

    def prepare_data(self, datasets):
        self.dataloaders = {split: DataLoader(dataset, batch_size=self.args.batch_size, num_workers=2)
                            for split, dataset in datasets.items()}

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def __getitem__(self, idx):
        return self.dataloaders['train'][idx]


class datainfos(AbstractDatasetInfos):
    def __init__(self, datamodule, args, info_type=None, info_num=None, recompute_statistics=False):
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'event'
        self.max_n_nodes = args.max_n
        if info_num != None:
            self.n_noded = info_num / args.max_n
        else:
            self.n_nodes = torch.ones(args.max_n, dtype=torch.float)/args.max_n
        if info_type != None:
            self.node_types = info_type / args.num_vertex_type
        else:
            self.node_types = torch.ones(args.num_vertex_type, dtype=torch.float)/args.num_vertex_type
        self.edge_types = torch.ones(args.num_edge_type, dtype=torch.float) / args.num_edge_type
        
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

