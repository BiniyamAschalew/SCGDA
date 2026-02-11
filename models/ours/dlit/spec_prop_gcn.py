"""
SpecPropGCN: Spectral Propagation GCN Layer

A GCN layer with spectral polynomial filters (Bernstein or Chebyshev) that supports
separate filters for source and target domains with shared feature projection weights.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from scipy.special import comb
from models.base_model import BaseGDA


class BernsteinFilter(MessagePassing):
    """
    Bernstein polynomial filter similar to BernProp from DGSDA.
    
    Computes: out = Σ_{i=0}^{K} θ_i * C(K,i)/2^K * (2I - L)^{K-i} * L^i * x
    """
    
    def __init__(self, K: int, learnable: bool = True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.K = K
        self.temp = nn.Parameter(torch.ones(K + 1), requires_grad=learnable)
        self._reset_parameters()
    
    def _reset_parameters(self):
        self.temp.data.fill_(1.0)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Apply Bernstein polynomial filter.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights
            
        Returns:
            Filtered features [N, F]
        """
        TEMP = F.relu(self.temp)
        num_nodes = x.size(0)
        
        # Compute normalized Laplacian: L = I - D^{-1/2}AD^{-1/2}
        edge_index_L, norm_L = get_laplacian(
            edge_index, edge_weight, 
            normalization='sym', 
            dtype=x.dtype,
            num_nodes=num_nodes
        )
        
        # Compute (2I - L) for propagation
        edge_index_diff, norm_diff = add_self_loops(
            edge_index_L, -norm_L, 
            fill_value=2.0, 
            num_nodes=num_nodes
        )
        
        # Pre-compute powers: (2I - L)^k * x for k=0,...,K
        tmp = [x]
        x_prop = x
        for _ in range(self.K):
            x_prop = self.propagate(edge_index_diff, x=x_prop, norm=norm_diff)
            tmp.append(x_prop)
        
        # Compute Bernstein basis combination
        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]
        
        for i in range(self.K):
            x_L = tmp[self.K - i - 1]
            # Apply L^{i+1}
            for _ in range(i + 1):
                x_L = self.propagate(edge_index_L, x=x_L, norm=norm_L)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x_L
        
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def get_energy(self, edge_index, num_nodes, device):
        """
        Compute filter energy response on unit signal.
        
        Returns:
            [N, 1] tensor of energy values at each node
        """
        ones_signal = torch.ones(num_nodes, 1, device=device)
        with torch.no_grad():
            energy = self.forward(ones_signal, edge_index)
        return energy


class ChebyshevFilter(MessagePassing):
    """
    Chebyshev polynomial filter.
    
    Computes: out = Σ_{k=0}^{K} θ_k * T_k(L̃) * x
    where L̃ = 2L/λ_max - I (scaled Laplacian)
    """
    
    def __init__(self, K: int, learnable: bool = True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.K = K
        self.temp = nn.Parameter(torch.ones(K + 1), requires_grad=learnable)
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize with uniform weights
        self.temp.data.fill_(1.0 / (self.K + 1))
    
    def forward(self, x, edge_index, edge_weight=None, lambda_max=2.0):
        """
        Apply Chebyshev polynomial filter.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights
            lambda_max: Maximum eigenvalue estimate (default 2.0)
            
        Returns:
            Filtered features [N, F]
        """
        TEMP = self.temp
        num_nodes = x.size(0)
        
        # Compute normalized Laplacian
        edge_index_L, norm_L = get_laplacian(
            edge_index, edge_weight,
            normalization='sym',
            dtype=x.dtype,
            num_nodes=num_nodes
        )
        
        # Scale to [-1, 1]: L̃ = 2L/λ_max - I
        # This is equivalent to using (2/λ_max) * L and adding -1 to self-loops
        scale = 2.0 / lambda_max
        norm_scaled = norm_L * scale
        # Adjust self-loops: subtract 1 from diagonal
        edge_index_scaled, norm_scaled = add_self_loops(
            edge_index_L, norm_scaled,
            fill_value=-1.0,
            num_nodes=num_nodes
        )
        
        # Chebyshev recurrence: T_0(x) = 1, T_1(x) = x, T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
        T_0 = x  # T_0(L̃) * x = x
        out = TEMP[0] * T_0
        
        if self.K >= 1:
            T_1 = self.propagate(edge_index_scaled, x=x, norm=norm_scaled)
            out = out + TEMP[1] * T_1
            
            T_prev, T_curr = T_0, T_1
            for k in range(2, self.K + 1):
                T_next = 2 * self.propagate(edge_index_scaled, x=T_curr, norm=norm_scaled) - T_prev
                out = out + TEMP[k] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def get_energy(self, edge_index, num_nodes, device):
        """
        Compute filter energy response on unit signal.
        """
        ones_signal = torch.ones(num_nodes, 1, device=device)
        with torch.no_grad():
            energy = self.forward(ones_signal, edge_index)
        return energy


def create_filter(filter_type: str, K: int, learnable: bool = True):
    """Factory function to create spectral filter."""
    filter_type = filter_type.lower()
    if filter_type == 'bernstein':
        return BernsteinFilter(K, learnable=learnable)
    elif filter_type == 'chebyshev':
        return ChebyshevFilter(K, learnable=learnable)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Use 'bernstein' or 'chebyshev'.")


class SpecPropGCN(BaseGDA):
    """
    Spectral Propagation GCN Layer.
    
    Features:
    - Separate spectral filters for source (Ks) and target (Kt) domains
    - Shared linear projection weights
    - KL divergence structural loss between filter energy outputs
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        Ks: Filter depth for source domain
        Kt: Filter depth for target domain
        filter_type: 'bernstein' or 'chebyshev'
        bias: Whether to use bias in linear layer
        dropout: Dropout probability
    """
    
    def __init__(self, config: dict):
        #     self, 
        #     in_dim: int, 
        #     out_dim: int, 
        #     Ks: int = 1, 
        #     Kt: int = 4, 
        #     filter_type: str = 'bernstein',
        #     bias: bool = True,
        #     dropout: float = 0.0
        # ):
        super(SpecPropGCN, self).__init__(config)
        
        # self.in_dim = config["model"]["in_dim"]
        # self.out_dim = config["model"]["out_dim"]
        self.Ks = config["model"]["Ks"]
        self.Kt = config["model"]["Kt"]
        self.filter_type = config["model"].get("filter_type", 'bernstein')
        self.dropout = config["model"]["dropout"]
        
        # Shared linear projection
        self.lin = nn.Linear(self.in_dim, self.out_dim)
        
        # Separate filters for source and target
        self.filter_s = create_filter(self.filter_type, self.Ks, learnable=True)
        self.filter_t = create_filter(self.filter_type, self.Kt, learnable=True)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        self.lin.reset_parameters()
        self.filter_s._reset_parameters()
        self.filter_t._reset_parameters()
    
    def forward(self, x, edge_index, edge_weight=None, is_source: bool = True):
        """
        Forward pass with domain-specific filter.
        
        Args:
            x: Input features [N, in_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights
            is_source: If True, use source filter; else target filter
            
        Returns:
            Filtered features [N, out_dim]
        """
        # Shared projection
        x = self.lin(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Domain-specific filtering
        if is_source:
            x = self.filter_s(x, edge_index, edge_weight)
        else:
            x = self.filter_t(x, edge_index, edge_weight)
        
        return x
    
    def struct_loss(
        self, 
        edge_index_s, 
        edge_index_t, 
        num_nodes_s: int, 
        num_nodes_t: int,
        device
    ):
        """
        Compute structural loss as KL divergence between filter energy outputs.
        
        Pass unit signal (all ones) through both filters and compute KL divergence
        between the energy distributions.
        
        Args:
            edge_index_s: Source domain edge indices
            edge_index_t: Target domain edge indices
            num_nodes_s: Number of nodes in source
            num_nodes_t: Number of nodes in target
            device: Torch device
            
        Returns:
            KL divergence loss (scalar)
        """
        # Get energy outputs from both filters
        energy_s = self.filter_s.get_energy(edge_index_s, num_nodes_s, device)
        energy_t = self.filter_t.get_energy(edge_index_t, num_nodes_t, device)
        
        # Normalize to create probability distributions
        # Use softmax to convert energy to probabilities
        prob_s = F.softmax(energy_s.squeeze(-1), dim=0)
        prob_t = F.softmax(energy_t.squeeze(-1), dim=0)
        
        # For KL divergence, we need same-sized distributions
        # Sample or interpolate to match sizes
        min_size = min(num_nodes_s, num_nodes_t)
        
        # Sample from both distributions to match sizes
        idx_s = torch.randperm(num_nodes_s, device=device)[:min_size]
        idx_t = torch.randperm(num_nodes_t, device=device)[:min_size]
        
        prob_s_sampled = prob_s[idx_s]
        prob_t_sampled = prob_t[idx_t]
        
        # Re-normalize after sampling
        prob_s_sampled = prob_s_sampled / prob_s_sampled.sum()
        prob_t_sampled = prob_t_sampled / prob_t_sampled.sum()
        
        # Add small epsilon for numerical stability
        eps = 1e-8
        prob_s_sampled = prob_s_sampled + eps
        prob_t_sampled = prob_t_sampled + eps
        
        # KL divergence: KL(P_s || P_t)
        kl_div = torch.sum(prob_s_sampled * (torch.log(prob_s_sampled) - torch.log(prob_t_sampled)))
        
        return kl_div
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_dim={self.in_dim}, out_dim={self.out_dim}, '
                f'Ks={self.Ks}, Kt={self.Kt}, filter={self.filter_type})')
