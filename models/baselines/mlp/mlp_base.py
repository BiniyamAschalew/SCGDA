import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Sequential, Linear
from torch_geometric.nn import global_mean_pool

from  models.__layers.build_layer import build_activation

class MLPBase(nn.Module):

    def __init__(self, config: dict): 
        super(MLPBase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]

        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]

        self.act_type = config["model"]["activation"]

        self.act = build_activation(self.act_type)
        self.layers = nn.ModuleList()

        # build the backbone gnn
        self.layers.append(Linear(self.in_dim, self.hid_dim))
        for i in range(self.num_layers-1):
            self.layers.append(Linear(self.hid_dim, self.hid_dim))

        self.cls = Linear(self.hid_dim, self.num_classes)
            
    def forward(self, x, edge_index, edge_weight=None, batch=None):

        x = self.feat_bottleneck(x) # edge_index, edge_weight, batch)
        x = self.feat_classifier(x) #, edge_index, edge_weight) 

        x = F.log_softmax(x, dim=1)

        return x
    
    def feat_bottleneck(self, x):

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    
    def feat_classifier(self, x):

        x = self.cls(x)
        return x
