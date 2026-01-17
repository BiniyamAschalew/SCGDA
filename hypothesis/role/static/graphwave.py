"""
GraphWave structural role encoding using PyTorch Geometric.

This module computes structural fingerprints for nodes using the characteristic
function of heat diffusion wavelets, as described in:
"Learning Structural Node Embeddings via Diffusion Wavelets" (KDD 2018)

Uses Chebyshev polynomial approximation for efficient heat kernel computation.
"""

from typing import Sequence, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_torch_csr_tensor


class GraphWave:
    """
    Compute structural roles using GraphWave algorithm with Chebyshev approximation.
    
    Each node's fingerprint is the empirical characteristic function of its
    heat wavelet coefficients, sampled at multiple time points.
    """

    def __init__(
        self,
        # scales: Sequence[float] | None = None,
        # time_points: Sequence[float] | None = None,
        # order: int = 30,
        # normalize: bool = True,
        config: dict,
    ):
        """
        Args:
            scales: Diffusion scales (tau values). Default: [1.0, 10.0, 25.0, 50.0]
            time_points: Points at which to sample characteristic function. 
                         Default: linspace(0, 2*pi, 25)
            order: Order of Chebyshev polynomial approximation. Default: 30
            normalize: Whether to z-normalize the output features.
        """

        scales = config["model"].get("graphwave_scales", None)
        time_points = config["model"].get("graphwave_time_points", None)
        order = config["model"].get("graphwave_chebyshev_order", 30)
        normalize = config["model"].get("graphwave_normalize", True)
        
        self.scales = list(scales) if scales is not None else [1.0, 10.0, 25.0, 50.0]
        if time_points is not None:
            self.time_points = torch.tensor(time_points, dtype=torch.float32)
        else:
            self.time_points = torch.linspace(0, 2 * torch.pi, 25)
        self.order = order
        self.normalize = normalize

    def _chebyshev_coefficients(self, scale: float) -> torch.Tensor:
        """
        Compute Chebyshev coefficients for exp(-scale * (x+1)) on [-1, 1].
        
        This approximates the heat kernel exp(-scale * L) when L's eigenvalues 
        are shifted to [-1, 1] via (L - I).
        """
        order = self.order
        # Chebyshev nodes
        k = torch.arange(1, order + 1, dtype=torch.float32)
        xx = torch.cos((2 * k - 1) / (2 * order) * torch.pi)
        
        # Build Chebyshev basis at these nodes
        basis = [torch.ones(order), xx]
        for _ in range(order - 1):
            basis.append(2 * xx * basis[-1] - basis[-2])
        basis = torch.stack(basis)  # (order+1, order)
        
        # Function values at Chebyshev nodes
        f = torch.exp(-scale * (xx + 1))
        
        # Compute coefficients via discrete Chebyshev transform
        products = f.unsqueeze(0) * basis  # (order+1, order)
        coeffs = (2.0 / order) * products.sum(dim=1)
        coeffs[0] = coeffs[0] / 2
        
        return coeffs

    def _compute_heat_kernel_chebyshev(self, laplacian: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Compute heat kernel using Chebyshev polynomial approximation.
        
        H(s) ≈ Σ_k c_k T_k(L - I)
        
        where T_k are Chebyshev polynomials and c_k are coefficients.
        """
        n = laplacian.shape[0]
        coeffs = self._chebyshev_coefficients(scale)
        
        # Shifted Laplacian: L_shifted = L - I (maps eigenvalues from [0,2] to [-1,1])
        lap_shifted = laplacian - torch.eye(n, dtype=laplacian.dtype, device=laplacian.device)
        
        # Build Chebyshev polynomial basis: T_0(L), T_1(L), T_2(L), ...
        # T_0 = I, T_1 = L_shifted, T_k = 2*L_shifted*T_{k-1} - T_{k-2}
        identity = torch.eye(n, dtype=laplacian.dtype, device=laplacian.device)
        monomes = [identity, lap_shifted]
        
        for _ in range(2, self.order + 1):
            next_monome = 2 * lap_shifted @ monomes[-1] - monomes[-2]
            monomes.append(next_monome)
        
        # Compute heat kernel as weighted sum
        heat = coeffs[0] * monomes[0]
        for k in range(1, self.order + 1):
            heat = heat + coeffs[k] * monomes[k]
        
        # Threshold small values for efficiency
        threshold = 1e-4 / max(1, n)
        heat = torch.where(torch.abs(heat) > threshold, heat, torch.zeros_like(heat))
        
        return heat

    def _characteristic_function(self, heat_kernel: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical characteristic function of heat distribution for each node.
        
        φ_u(t) = (1/n) * Σ_v exp(i*t*H[u,v]) = (1/n) * Σ_v [cos(t*H[u,v]) + i*sin(t*H[u,v])]
        """
        coeffs = []
        for t in self.time_points:
            cos_vals = torch.cos(t * heat_kernel).mean(dim=1)
            sin_vals = torch.sin(t * heat_kernel).mean(dim=1)
            coeffs.extend([cos_vals, sin_vals])
        
        return torch.stack(coeffs, dim=1)

    def _encode(self, data: Data) -> torch.Tensor:
        """Encode structural roles for a single graph."""
        num_nodes = data.num_nodes
        
        if num_nodes == 0:
            return torch.zeros(0, len(self.scales) * 2 * len(self.time_points))
        
        # Get normalized Laplacian
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            normalization='sym',
            num_nodes=num_nodes
        )
        
        # Convert to dense (needed for Chebyshev polynomial computation)
        laplacian = torch.zeros(num_nodes, num_nodes)
        laplacian = laplacian.to(data.edge_index.device)
        laplacian[edge_index[0], edge_index[1]] = edge_weight
        
        # Compute characteristic function for each scale
        all_features = []
        for scale in self.scales:
            heat = self._compute_heat_kernel_chebyshev(laplacian, scale)
            chi = self._characteristic_function(heat)
            all_features.append(chi)
        
        roles = torch.cat(all_features, dim=1)
        
        if self.normalize and roles.numel() > 0:
            mean = roles.mean(dim=0, keepdim=True)
            std = roles.std(dim=0, keepdim=True).clamp(min=1e-6)
            roles = (roles - mean) / std
        
        return roles

    def build(self, source_data: Data, target_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build role encodings for source and target graphs."""
        src_roles = self._encode(source_data)
        tgt_roles = self._encode(target_data)
        return src_roles, tgt_roles
