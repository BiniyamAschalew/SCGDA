import os
import os.path as osp
import csv
import json

import numpy as np
import scipy.sparse 
import warnings

import torch
import numpy as np
from torch_geometric.data import Data
import scipy.io as sio

from Learn.Clean_SCGDA.data.base_dataset import BaseDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)


class BlogDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=None,
    ):
        super(BlogDataset, self).__init__(
            name, config, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    def load_dataset(self):
        path = osp.join(self.raw_dir, "{}.mat".format(self.name))
        net = sio.loadmat(path)
        X, A, Y = net["attrb"], net["network"], net["group"]
        Y = np.argmax(Y, axis=1)
        return X, A, Y


    def process(self):
        features, A, label = self.load_dataset()
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

        x = torch.from_numpy(features.astype(int)).to(torch.float)
        y = torch.from_numpy(label).to(torch.int64)

        data_list = []

        train_masks, val_masks, test_masks = self.create_masks(y)

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
        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])



