from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.utils import degree


class RandomRole:
    """
    Random role as a baseline structural role encoding.
    """

    def __init__(self, config: dict):
        
        self.steps = max(1, config["model"]["role_dim"])



    def _encode(self, data: Data) -> torch.Tensor:

        num_nodes = data.num_nodes
        trajectory = torch.randn(num_nodes, self.steps, dtype=torch.float32, device=data.edge_index.device)
        return trajectory
        
        
    def build(self, source_data: Data, target_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        src_roles = self._encode(source_data)
        tgt_roles = self._encode(target_data)
        
        return src_roles, tgt_roles

    