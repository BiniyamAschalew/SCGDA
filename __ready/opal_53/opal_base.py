from torch import nn
import torch.nn.functional as F
from Learn.Clean_SCGDA.models.ours.opal.chebprop import ChebProp



class OPALBase(nn.Module):

    def __init__(self, num_layers, features, hidden, classes, 
                 dropout_rate=0.0, K=15, cheb_lambda_max=2.0):
        super(OPALBase, self).__init__()

        self.src_filter = ChebProp(K=K)
        self.tgt_filter = ChebProp(K=K)
        self.cls_filter = ChebProp(K=K)

        self.encoder = nn.ModuleList()
        self.classifier = nn.ModuleList()

        self.encoder.append(nn.Linear(features, hidden))
        for _ in range(num_layers-2):
            self.encoder.append(nn.Linear(hidden, hidden))

        self.classifier.append(nn.Linear(hidden, classes))

        self.dropout_rate = dropout_rate
        self.cheb_lambda_max = cheb_lambda_max

    def reset_parameters(self):
        self.src_filter.reset_parameters()
        self.tgt_filter.reset_parameters()
        self.cls_filter.reset_parameters()

    def feat_bottleneck(self, x, edge_index, is_source_domain=True):

        filter = self.src_filter
        if not is_source_domain:
            filter = self.tgt_filter

        for lin in self.encoder:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.propagate(filter, x, edge_index)

        return x
    
    
    def feat_classifier(self, x, edge_index):

        filter = self.cls_filter
        for lin in self.classifier:

            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = lin(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.propagate(filter, x, edge_index)

        return x
    
    def project(self, x):
        first_lin = self.encoder[0]
        return first_lin(x)

    def forward(self, data, is_source_domain=True):

        x, edge_index = data.x, data.edge_index

        h = self.feat_bottleneck(x, edge_index, is_source_domain)
        y = self.feat_classifier(h, edge_index)
        return y

    def propagate(self, prop, x, edge_index):
        return prop(x, edge_index, lambda_max=self.cheb_lambda_max)
