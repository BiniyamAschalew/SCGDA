import os
import os.path as osp
import warnings

import torch
import numpy as np
from torch_geometric.data import Data
import scipy.io as sio

from data.base_dataset import BaseDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)


class BlogDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super(BlogDataset, self).__init__(
            name, config, transform, pre_transform, pre_filter
        )

        self.data, self.slices = torch.load(self.processed_paths[0])


