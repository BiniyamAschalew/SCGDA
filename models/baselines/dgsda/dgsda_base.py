from torch import nn
import torch.nn.functional as F

from models.__components.bernprop import BernProp



class DGSDABase(nn.Module):

    def __init__(self, features, hidden, classes, dprate=0.0, K=15):
        super(DGSDABase, self).__init__()
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)
        self.prop1 = BernProp(K)
        self.prop2 = BernProp(K)
        self.prop3 = BernProp(K)

        self.dprate = dprate

    def reset_parameters(self):

        self.prop1.reset_parameters()

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
