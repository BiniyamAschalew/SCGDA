from __future__ import annotations

import torch
from torch import nn

from Learn.Clean_SCGDA.models.baselines.gnn.gnn_base import GNNBase


class TripletSimGDAModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.backbone = GNNBase(config)

    def encode(self, data, x: torch.Tensor | None = None) -> torch.Tensor:
        features = data.x if x is None else x
        return self.backbone.feat_bottleneck(
            features,
            data.edge_index,
            getattr(data, "edge_weight", None),
            getattr(data, "batch", None),
        )

    def classify(self, features: torch.Tensor, data) -> torch.Tensor:
        return self.backbone.feat_classifier(
            features,
            data.edge_index,
            getattr(data, "edge_weight", None),
        )

    def forward(self, data, x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(data, x=x)
        logits = self.classify(features, data)
        return logits, features
