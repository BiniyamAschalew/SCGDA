import torch
from torch import nn
import torch.nn.functional as F

from  models.__components.bernprop import BernProp


class BDliteBase(nn.Module):
    """
    BernNet variant matching the original DLITE architecture.

    Exposes:
    1) linear encoder feature space before graph filtering
    2) filter-only forward for source/target Bernstein filters
    """

    def __init__(self, features, hidden, classes, dropout, dprate=0.0, K=15):
        super(BDliteBase, self).__init__()
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)

        # Source/target domain filters + classifier-side propagation
        self.prop1 = BernProp(K)  # source filter
        self.prop2 = BernProp(K)  # target filter
        self.prop3 = BernProp(K)  # classifier-side propagation

        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.prop2.reset_parameters()
        self.prop3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def encode_linear(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        return x

    def filter_only(self, x, edge_index, domain="source", apply_dropout=True):
        if apply_dropout:
            x = F.dropout(x, p=self.dprate, training=self.training)

        if domain == "source":
            return self.prop1(x, edge_index)
        if domain == "target":
            return self.prop2(x, edge_index)
        raise ValueError(f"Unknown domain: {domain}")

    def get_props(self, x, edge_index, is_source_domain=True):
        x = self.encode_linear(x)
        if is_source_domain:
            x = self.filter_only(x, edge_index, domain="source", apply_dropout=True)
        else:
            x = self.filter_only(x, edge_index, domain="target", apply_dropout=True)
        return x

    def forward(self, data, is_source_domain=True):
        x, edge_index = data.x, data.edge_index

        x = self.get_props(x, edge_index, is_source_domain=is_source_domain)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop3(x, edge_index)
        return x
