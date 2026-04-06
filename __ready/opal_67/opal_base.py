from torch import nn
import torch.nn.functional as F

# from models.__components.bernprop import BernProp
from Learn.Clean_SCGDA.models.ours.opal.chebprop import ChebProp



class OPALBase(nn.Module):

    def __init__(self, features, hidden, classes, dropout_rate=0.0, K=15, cheb_lambda_max=2.0):
        super(OPALBase, self).__init__()
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)


        self.src_filter = ChebProp(K=K)
        self.tgt_filter = ChebProp(K=K)
        self.cls_filter = ChebProp(K=K)

        self.dropout_rate = dropout_rate
        self.cheb_lambda_max = cheb_lambda_max

    def reset_parameters(self):

        self.src_filter.reset_parameters()

    def forward(self, data, is_source_domain=True):

        x, edge_index = data.x, data.edge_index

        x = self.get_props(x, edge_index, is_source_domain)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.propagate(self.cls_filter, x, edge_index)
        return x

    def get_props(self, x, edge_index, is_source_domain=True):

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if is_source_domain:
            filter = self.src_filter
        else:
            filter = self.tgt_filter

        x = self.propagate(filter, x, edge_index)
        return x

    def propagate(self, prop, x, edge_index):
        return prop(x, edge_index, lambda_max=self.cheb_lambda_max)
