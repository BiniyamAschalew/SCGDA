"""
SCGDABase: SCGDA Base model - closely matches DGSDABase structure.

Key differences from DGSDA:
- prop1 and prop2 can have different depths (Ks, Kt)
- Supports both Bernstein and Chebyshev filters
"""

from torch import nn
import torch.nn.functional as F

from models.__components.bernprop import BernProp


class SCGDABase(nn.Module):
    """
    SCGDA Base model following DGSDA architecture.
    
    Architecture: lin1 -> prop1/prop2 (S/T) -> lin2 -> prop3
    
    Key differences from DGSDA:
    - prop1 (source) has depth Ks
    - prop2 (target) has depth Kt
    - Supports shared prop3 depth K3
    """

    def __init__(self, features, hidden, classes, dprate=0.0, Ks=1, Kt=4, K3=None):
        super(SCGDABase, self).__init__()
        
        # Default K3 to max of Ks, Kt
        if K3 is None:
            K3 = max(Ks, Kt)
        
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)
        
        # Separate source/target filters with potentially different depths
        self.prop1 = BernProp(Ks)  # Source filter
        self.prop2 = BernProp(Kt)  # Target filter
        self.prop3 = BernProp(K3)  # Shared classifier filter

        self.dprate = dprate

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.prop2.reset_parameters()
        self.prop3.reset_parameters()

    def forward(self, data, is_source_domain=True):
        x, edge_index = data.x, data.edge_index

        x = self.get_props(x, edge_index, is_source_domain)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop3(x, edge_index)
        return x

    def get_props(self, x, edge_index, is_source_domain=True):
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=self.dprate, training=self.training)
        if is_source_domain:
            x = self.prop1(x, edge_index)
        else:
            x = self.prop2(x, edge_index)
        return x
    
    def energy_loss(self, source_edge_index, target_edge_index, num_nodes_s, num_nodes_t, device):
        """
        Compute KL divergence loss between filter energy responses on unit signal.
        
        Passes ones signal through both filters and computes KL divergence
        between the resulting energy distributions.
        """
        import torch
        
        # Create ones signals
        ones_s = torch.ones(num_nodes_s, 1, device=device)
        ones_t = torch.ones(num_nodes_t, 1, device=device)
        
        # Get energy outputs from both filters
        energy_s = self.prop1(ones_s, source_edge_index)
        energy_t = self.prop2(ones_t, target_edge_index)
        
        # Convert to probability distributions using softmax
        prob_s = F.softmax(energy_s.squeeze(-1), dim=0)
        prob_t = F.softmax(energy_t.squeeze(-1), dim=0)
        
        # Sample to match sizes for KL divergence
        min_size = min(num_nodes_s, num_nodes_t)
        idx_s = torch.randperm(num_nodes_s, device=device)[:min_size]
        idx_t = torch.randperm(num_nodes_t, device=device)[:min_size]
        
        prob_s_sampled = prob_s[idx_s]
        prob_t_sampled = prob_t[idx_t]
        
        # Re-normalize after sampling
        prob_s_sampled = prob_s_sampled / prob_s_sampled.sum()
        prob_t_sampled = prob_t_sampled / prob_t_sampled.sum()
        
        # Add epsilon for numerical stability
        eps = 1e-8
        prob_s_sampled = prob_s_sampled + eps
        prob_t_sampled = prob_t_sampled + eps
        
        # KL divergence: KL(P_s || P_t)
        kl_div = torch.sum(prob_s_sampled * (torch.log(prob_s_sampled) - torch.log(prob_t_sampled)))
        
        return kl_div
