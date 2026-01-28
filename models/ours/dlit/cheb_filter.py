"""
Chebyshev polynomial filter for spectral graph operations.

Provides:
- Learnable Chebyshev filter coefficients
- Role computation via repeated filter application
- Graph Laplacian utilities
"""

import torch
from torch import nn
from torch_geometric.utils import get_laplacian, add_self_loops, degree


class ChebFilter(nn.Module):
    """
    Learnable Chebyshev polynomial filter.
    
    Applies: g(L) = Σ_{k=0}^{K-1} θ_k T_k(L̃)
    where T_k are Chebyshev polynomials and L̃ is the scaled Laplacian.
    """

    def __init__(self, K: int = 5, normalization: str = "sym"):
        super().__init__()
        self.K = K
        self.normalization = normalization
        
        # Learnable filter coefficients
        self.theta = nn.Parameter(torch.ones(K) / K)
        
        # Cache for Laplacian
        self._cached_edge_index = None
        self._cached_edge_weight = None
        self._cached_num_nodes = None

    def _get_laplacian(self, edge_index, num_nodes, dtype):
        """Get normalized Laplacian, with caching."""
        if (self._cached_num_nodes == num_nodes and 
            self._cached_edge_index is not None and
            self._cached_edge_index.device == edge_index.device):
            return self._cached_edge_index, self._cached_edge_weight
        
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index,
            normalization=self.normalization,
            dtype=dtype,
            num_nodes=num_nodes,
        )
        
        self._cached_edge_index = edge_index_lap
        self._cached_edge_weight = edge_weight_lap
        self._cached_num_nodes = num_nodes
        
        return edge_index_lap, edge_weight_lap

    def _cheb_mv(self, edge_index, edge_weight, x, lambda_max=2.0):
        """
        Chebyshev recurrence matrix-vector multiplication.
        Computes: T_k(L̃) x where L̃ = 2L/λ_max - I
        """
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * edge_weight.unsqueeze(-1))
        return (2.0 / lambda_max) * out - x

    def forward(self, x, edge_index, lambda_max=2.0):
        """
        Apply Chebyshev filter to input signal.
        
        Args:
            x: Input signal [N, F]
            edge_index: Graph connectivity
            lambda_max: Maximum eigenvalue estimate
            
        Returns:
            Filtered signal [N, F]
        """
        num_nodes = x.size(0)
        edge_index_lap, edge_weight_lap = self._get_laplacian(
            edge_index, num_nodes, x.dtype
        )
        
        # T_0(L̃) x = x
        t0 = x
        out = self.theta[0] * t0
        
        if self.K == 1:
            return out
        
        # T_1(L̃) x = L̃ x
        t1 = self._cheb_mv(edge_index_lap, edge_weight_lap, t0, lambda_max)
        out = out + self.theta[1] * t1
        
        # T_k(L̃) = 2 L̃ T_{k-1}(L̃) - T_{k-2}(L̃)
        for k in range(2, self.K):
            t2 = 2.0 * self._cheb_mv(edge_index_lap, edge_weight_lap, t1, lambda_max) - t0
            out = out + self.theta[k] * t2
            t0, t1 = t1, t2
        
        return out

    def compute_roles(self, edge_index, num_nodes, steps: int, dtype=torch.float32):
        """
        Compute spectral structural roles via repeated filter application.
        
        Starting from a constant signal (all ones), applies the learned filter
        iteratively to produce a K-dimensional role embedding per node.
        
        Args:
            edge_index: Graph connectivity
            num_nodes: Number of nodes
            steps: Number of filter applications (role dimension)
            dtype: Data type for computation
            
        Returns:
            Role embeddings [N, steps]
        """
        device = edge_index.device
        edge_index_lap, edge_weight_lap = self._get_laplacian(
            edge_index, num_nodes, dtype
        )
        
        # Start with constant signal
        x = torch.ones((num_nodes, 1), device=device, dtype=dtype)
        roles = []
        
        for _ in range(steps):
            # Apply filter (detached from feature transformations)
            x = self.forward(x, edge_index, lambda_max=2.0)
            roles.append(x)
        
        return torch.cat(roles, dim=1)

    def clear_cache(self):
        """Clear cached Laplacian."""
        self._cached_edge_index = None
        self._cached_edge_weight = None
        self._cached_num_nodes = None
