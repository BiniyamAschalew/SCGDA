"""
DLIT Encoder: Chebyshev-based feature encoder for DLIT.

Uses ChebConv layers to learn graph representations that are
then used for classification and domain alignment.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from  models.__layers.build_layer import build_activation


class DLITEncoder(nn.Module):
    """
    Chebyshev-based encoder for DLIT.
    
    Architecture:
    - Multiple ChebConv layers for feature extraction (feat_bottleneck)
    - Final ChebConv layer for classification (feat_classifier)
    """

    def __init__(self, config: dict):
        super().__init__()
        
        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        
        # Chebyshev filter parameters
        self.cheb_K = config["model"].get("cheb_K", 5)
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
        self.classifier = ChebConv(
            self.hid_dim,
            self.num_classes,
            K=self.cheb_K,
            normalization=self.cheb_norm,
            bias=self.cheb_bias,
        )

    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        """
        Extract features through ChebConv layers.
        
        Args:
            x: Node features [N, F]
            edge_index: Graph connectivity
            edge_weight: Optional edge weights
            
        Returns:
            Node embeddings [N, hid_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feat_classifier(self, x, edge_index, edge_weight=None):
        """
        Classify node embeddings.
        
        Args:
            x: Node embeddings [N, hid_dim]
            edge_index: Graph connectivity
            edge_weight: Optional edge weights
            
        Returns:
            Class logits [N, num_classes]
        """
        return self.classifier(x, edge_index, edge_weight=edge_weight)

    def forward(self, x, edge_index, edge_weight=None):
        """Full forward pass: features -> classification."""
        h = self.feat_bottleneck(x, edge_index, edge_weight)
        logits = self.feat_classifier(h, edge_index, edge_weight)
        return F.log_softmax(logits, dim=1)
