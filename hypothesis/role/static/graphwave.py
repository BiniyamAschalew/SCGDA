"""
GraphWave-style structural role encoding using PyTorch Geometric.

This module computes structural fingerprints for nodes using the graph heat kernel.
Each node is represented by its diagonal heat values across multiple diffusion scales.

Reference: "Learning Structural Node Embeddings via Diffusion Wavelets" (KDD 2018)
"""

from typing import Sequence, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj


class GraphWave:
    """
    Compute structural roles from the graph heat kernel (GraphWave-style).
    
    Each node's fingerprint is the diagonal of exp(-scale * L) for multiple scales,
    where L is the normalized graph Laplacian.
    """

    def __init__(
        self,
        scales: Sequence[float] | None = None,
        normalize: bool = True,
    ):
        """
        Args:
            scales: Diffusion scales to use. Default: [1.0, 5.0, 10.0]
            normalize: Whether to z-normalize the output features.
        """
        self.scales = list(scales) if scales is not None else [1.0, 5.0, 10.0]
        self.normalize = normalize

    def _compute_heat_diagonals(self, laplacian: torch.Tensor) -> torch.Tensor:
        """
        Compute heat kernel diagonals for all scales using eigendecomposition.
        
        Heat kernel: H(s) = exp(-s * L) = V * exp(-s * Λ) * V^T
        Diagonal: H(s)_{ii} = sum_k (V_{ik})^2 * exp(-s * λ_k)
        
        Args:
            laplacian: Dense Laplacian matrix (n x n)
            
        Returns:
            Tensor of shape (n, num_scales) with diagonal heat values
        """
        # Eigendecomposition of symmetric Laplacian
        eigvals, eigvecs = torch.linalg.eigh(laplacian)
        
        # Compute diagonals for each scale: diag(V @ diag(exp(-s*λ)) @ V.T)
        # Simplified: sum over k of V_{ik}^2 * exp(-s * λ_k)
        eigvecs_sq = eigvecs ** 2  # (n, n)
        
        diagonals = []
        for scale in self.scales:
            exp_eigvals = torch.exp(-scale * eigvals)  # (n,)
            diag = eigvecs_sq @ exp_eigvals  # (n,)
            diagonals.append(diag.unsqueeze(1))
        
        return torch.cat(diagonals, dim=1)

    def _encode(self, data: Data) -> torch.Tensor:
        """
        Encode structural roles for a single graph.
        
        Args:
            data: PyG Data object
            
        Returns:
            Tensor of shape (num_nodes, num_scales) representing role fingerprints
        """
        num_nodes = data.num_nodes
        
        if num_nodes == 0:
            return torch.zeros(0, len(self.scales))
        
        # Get normalized Laplacian using PyG utility
        edge_index, edge_weight = get_laplacian(
            data.edge_index, 
            normalization='sym',
            num_nodes=num_nodes
        )
        
        # Convert to dense for eigendecomposition
        laplacian = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)
        laplacian = laplacian.squeeze(0)  # (n, n)
        
        # Compute heat kernel diagonals
        roles = self._compute_heat_diagonals(laplacian)
        
        # Optional normalization
        if self.normalize and roles.numel() > 0:
            mean = roles.mean(dim=0, keepdim=True)
            std = roles.std(dim=0, keepdim=True).clamp(min=1e-6)
            roles = (roles - mean) / std
        
        return roles

    def build(self, source_data: Data, target_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build role encodings for source and target graphs.
        
        Args:
            source_data: Source graph
            target_data: Target graph
            
        Returns:
            Tuple of (source_roles, target_roles), each of shape (num_nodes, num_scales)
        """
        src_roles = self._encode(source_data)
        tgt_roles = self._encode(target_data)
        return src_roles, tgt_roles
