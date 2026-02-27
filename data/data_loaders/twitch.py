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


class TwitchDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        config: dict,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=None,
    ):
        super(TwitchDataset, self).__init__(
            name, config, transform, pre_transform, pre_filter, force_reload=force_reload
        )


        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # twich has domain specifc naming for raw files

        domain = self.name
        raw_files = [
            f"musae_{domain}_edges.csv",
            f"musae_{domain}_features.json",
            f"musae_{domain}_target.csv",
        ]
        return raw_files



    def load_dataset(self):
        filepath = self.raw_dir
        label = []
        node_ids = []
        src = []
        targ = []
        uniq_ids = set()

        target_path = osp.join(filepath, f"musae_{self.name}_target.csv")
        with open(target_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node_id = int(row[5])
                if node_id not in uniq_ids:
                    uniq_ids.add(node_id)
                    label.append(int(row[2] == "True"))
                    node_ids.append(node_id)

        node_ids = np.array(node_ids, dtype=int)

        edges_path = osp.join(filepath, f"musae_{self.name}_edges.csv")
        with open(edges_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                src.append(int(row[0]))
                targ.append(int(row[1]))

        features_path = osp.join(filepath, f"musae_{self.name}_features.json")
        with open(features_path, "r") as f:
            features_json = json.load(f)

        src = np.array(src, dtype=int)
        targ = np.array(targ, dtype=int)
        label = np.array(label, dtype=int)

        inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
        reorder_node_ids = np.zeros_like(node_ids)
        for i in range(label.shape[0]):
            reorder_node_ids[i] = inv_node_ids[i]

        n = label.shape[0]
        A = scipy.sparse.csr_matrix(
            (np.ones(len(src)), (src, targ)), shape=(n, n)
        )
        features = np.zeros((n, 3170))
        for node, feats in features_json.items():
            node_idx = int(node)
            if node_idx >= n:
                continue
            features[node_idx, np.array(feats, dtype=int)] = 1

        label = label[reorder_node_ids]

        return features, A, label


    def process(self):

        features, A, label = self.load_dataset()
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        x = torch.from_numpy(features).to(torch.float)
        y = torch.from_numpy(label).to(torch.int64)

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


