import os.path as osp
import warnings

import torch
import numpy as np

from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_txt_array

warnings.filterwarnings("ignore", category=DeprecationWarning)


class BaseDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        self.name = name
        self.config = config
        self.root = osp.join(config["data"]["root"], name)

        super(BaseDataset, self).__init__(
            self.root, transform, pre_transform, pre_filter
        )

    @property
    def raw_file_names(self):
        return self.config["data"]["raw_file_names"]

    @property
    def processed_file_names(self):
        return self.config["data"]["processed_file_names"]

    def download(self):
        pass

    def process(self):
        pass

    def load_dataset(self):
        pass

    def load_labels(self):
        label_path = osp.join(self.raw_dir, f"{self.name}_labels.txt")
        f = open(label_path, "rb")
        content_list = []

        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)

        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

        return y

    def load_edge_index(self):
        edge_path = osp.join(self.raw_dir, f"{self.name}_edgelist.txt")
        edge_index = read_txt_array(edge_path, sep=",", dtype=torch.long).t()
        return edge_index

    def load_docs(self):
        docs_path = osp.join(self.raw_dir, f"{self.name}_docs.txt")
        content_list = []

        with open(docs_path, "rb") as f:
            lines = f.readlines()

        for line in lines:
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))

        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        return x

    def create_masks(self, y):
        num_data = y.shape[0]
        random_node_indices = np.random.permutation(num_data)

        training_size = int(num_data * self.config["data"]["train_split"])
        val_size = int(num_data * self.config["data"]["val_split"])

        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size : training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size :]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        return train_masks, val_masks, test_masks





