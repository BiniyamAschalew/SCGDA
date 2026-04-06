import os
import os.path as osp
import csv
import json

import numpy as np
import scipy.sparse

import warnings

import torch
from torch_geometric.data import Data

from Learn.Clean_SCGDA.data.base_dataset import BaseDataset


class AirportDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=None,
    ):
        super(AirportDataset, self).__init__(
            name, config, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.data, self.slices = torch.load(self.processed_paths[0])



    def process(self):
        edge_index = self.load_edge_index()
        y = self.load_labels()
        # for airport, we don't have node features

        data_list = []

        train_masks, val_masks, test_masks = self.create_masks(y)
        data = Data(
            edge_index=edge_index,
            x=None,
            y=y,
            num_nodes=y.size(0),
            train_mask=train_masks,
            val_mask=val_masks,
            test_mask=test_masks,
        )
        data = self.apply_pre_transform(data)

        data_list.append(data)
        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])



