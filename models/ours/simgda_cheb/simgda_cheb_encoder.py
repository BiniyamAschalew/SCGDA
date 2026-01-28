"""
Chebyshev-based encoder for SimGDACheb.

Uses ChebConv layers instead of standard GCN for spectral filtering.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool

from models.__layers.build_layer import build_activation


class SimGDAChebEncoder(nn.Module):
    """
    Chebyshev-based encoder for SimGDA.
    
    Replaces standard GCN with ChebConv layers for better spectral control.
    """

    def __init__(self, config: dict):
        super().__init__()
        
        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]
        
        # Chebyshev filter parameters
        self.cheb_K = config["model"].get("cheb_K", 3)
        self.cheb_norm = config["model"].get("cheb_norm", "sym")
        self.cheb_bias = config["model"].get("cheb_bias", True)
        
        self.act = build_activation(config["model"]["activation"])
        
        # Feature extraction layers
        self.convs = nn.ModuleList()
        self.convs.append(
            ChebConv(
                self.in_dim,
                self.hid_dim,
                K=self.cheb_K,
                normalization=self.cheb_norm,
                bias=self.cheb_bias,
            )
        )
        for _ in range(self.num_layers - 1):
            self.convs.append(
                ChebConv(
                    self.hid_dim,
                    self.hid_dim,
                    K=self.cheb_K,
                    normalization=self.cheb_norm,
                    bias=self.cheb_bias,
                )
            )
        
        # Classification layer
        if self.mode == "node":
            self.cls = ChebConv(
                self.hid_dim,
                self.num_classes,
                K=self.cheb_K,
                normalization=self.cheb_norm,
                bias=self.cheb_bias,
            )
        else:
            self.cls = nn.Linear(self.hid_dim, self.num_classes)

    def feat_bottleneck(self, x, edge_index, edge_weight=None, batch=None):
        """Extract features through ChebConv layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.mode == "graph":
            x = global_mean_pool(x, batch)
        
        return x

    def feat_classifier(self, x, edge_index, edge_weight=None):
        """Classify node embeddings."""
        if self.mode == "node":
            return self.cls(x, edge_index, edge_weight=edge_weight)
        return self.cls(x)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """Full forward pass."""
        h = self.feat_bottleneck(x, edge_index, edge_weight, batch)
        logits = self.feat_classifier(h, edge_index, edge_weight)
        return F.log_softmax(logits, dim=1)
