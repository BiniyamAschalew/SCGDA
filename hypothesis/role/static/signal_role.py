from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.utils import degree


class SignalRole:
    """
    Compute structural roles based on how a uniform signal evolves under repeated
    GCN-style propagation.
    """

    def __init__(self, steps: int = 8, operator: str = "sym_norm"):
        
        self.steps = max(1, steps)
        self.operator = operator
        self.conv = LGConv(normalize=(operator == "sym_norm"))
    


    def _encode(self, data: Data) -> torch.Tensor:
        """
        Encode structural roles for a single graph.
        
        Args:
            data: PyG Data object with edge_index and num_nodes
            
        Returns:
            Tensor of shape (num_nodes, steps) representing role fingerprints
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Compute edge weights for random walk normalization: D^{-1}
        # For symmetric, LGConv handles normalization internally
        edge_weight = None
        if self.operator == 'random_walk':
            row = edge_index[0]
            deg = degree(row, num_nodes, dtype=torch.float32)

            # handling cases where degree is zero
            deg_inv = torch.zeros_like(deg)
            mask = deg > 0

            deg_inv[mask] = 1.0 / deg[mask]
            edge_weight = deg_inv[row]
        
        # Initialize uniform signal
        signal = torch.ones(num_nodes, 1, dtype=torch.float32)
        trajectory = []

        # Propagate signal K times, collecting the trajectory
        for _ in range(self.steps):
            signal = self.conv(signal, edge_index, edge_weight)
            trajectory.append(signal)

        return torch.cat(trajectory, dim=1)

    def build(self, source_data: Data, target_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build role encodings for source and target graphs.
        
        Args:
            source_data: Source graph
            target_data: Target graph
            
        Returns:
            Tuple of (source_roles, target_roles), each of shape (num_nodes, steps)
        """
        src_roles = self._encode(source_data)
        tgt_roles = self._encode(target_data)
        
        return src_roles, tgt_roles

    