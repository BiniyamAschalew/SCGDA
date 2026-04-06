import os
import os.path as osp
import csv
import json

import numpy as np
import scipy.sparse
import warnings

import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array

from  data.base_dataset import BaseDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MAGDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=None,
    ):
        super(MAGDataset, self).__init__(
            name, config, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):

        path = osp.join(self.raw_dir, '{}_labels_20.pt'.format(self.name))
        graph = torch.load(path)
        x, edge_index, y = graph.x, graph.edge_index, graph.y

        train_masks, val_masks, test_masks = self.create_masks(y)

        data_list = []
        data = Data(
            edge_index=edge_index,
            x=x,
            y=y,
            num_nodes=y.size(0),
            train_mask=train_masks,
            val_mask=val_masks,
            test_mask=test_masks,
        )
        data = self.apply_pre_transform(data)

        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])






