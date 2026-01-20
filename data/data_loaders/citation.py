import os
import os.path as osp
import csv
import json

import numpy as np
import scipy.sparse

import warnings

import torch
from torch_geometric.data import Data

from data.base_dataset import BaseDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)


class CitationDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=None,
    ):
        super(CitationDataset, self).__init__(
            name, config, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):

        edge_index = self.load_edge_index()
        x = self.load_docs()
        y = self.load_labels()

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

        if self.pre_transform is not None:
            if not os.path.exists(self.processed_paths[0] + "eival.pt"):
                data = self.pre_transform(data, self.processed_paths[0])

        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


