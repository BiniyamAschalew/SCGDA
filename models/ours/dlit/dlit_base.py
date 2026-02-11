from torch import nn
import torch.nn.functional as F

from models.__components.bernprop import BernProp


class DLITBase(nn.Module):
    """DLIT backbone with one shared Bernstein filter."""

    def __init__(self, features, hidden, classes, K=8, dprate=0.0):
        super().__init__()
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)
        self.prop = BernProp(K)
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data, is_source_domain=True):
        x, edge_index = data.x, data.edge_index

        x = self.get_props(x, edge_index)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop(x, edge_index)
        return x

    def get_props(self, x, edge_index):
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=self.dprate, training=self.training)
        return self.prop(x, edge_index)
