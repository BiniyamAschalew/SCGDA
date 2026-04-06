import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Sequential, Linear
from torch_geometric.nn import global_mean_pool

from Learn.Clean_SCGDA.models.__layers.build_layer import build_layer, build_activation

class FDABase(nn.Module):

    def __init__(self, config: dict): 
        super(FDABase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]

        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]

        self.gnn_type = config["model"]["gnn"]  
        self.act_type = config["model"]["activation"]
        self.mode = config["model"]["mode"]

        self.act = build_activation(self.act_type)
        self.convs = nn.ModuleList()

        # build the backbone gnn
        self.convs.append(build_layer(self.in_dim, self.hid_dim, self.gnn_type))
        for i in range(self.num_layers-1):
            self.convs.append(build_layer(self.hid_dim, self.hid_dim, self.gnn_type))

        if self.mode == 'node':
            self.cls = build_layer(self.hid_dim, self.num_classes, self.gnn_type)

        elif self.mode == 'graph':
            self.cls = Linear(self.hid_dim, self.num_classes)
            
    def forward(self, x, edge_index, edge_weight=None, batch=None):

        x = self.feat_bottleneck(x, edge_index, edge_weight, batch)
        x = self.feat_classifier(x, edge_index, edge_weight) 

        x = F.log_softmax(x, dim=1)

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_weight=None, batch=None):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == 'graph':
            x = global_mean_pool(x, batch)

        return x
    
    def feat_classifier(self, x, edge_index, edge_weight=None):

        if self.mode == 'node':
            x = self.cls(x, edge_index, edge_weight)
        else:
            x = self.cls(x)
        
        return x
