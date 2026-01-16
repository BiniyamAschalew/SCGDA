import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from models.__layers.build_layer import build_layer, build_activation
from models.__layers.reverse_layer import GradReverse



class A2GNNBase(nn.Module):

    def __init__(self, config: dict):
        super(A2GNNBase, self).__init__()
        
        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]

        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.adv = config["model"]["adv"]

        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]
        self.act = build_activation(config["model"]["activation"])
        self.gnn_type = config["model"]["gnn"]  

        self.convs = nn.ModuleList()

        self.convs.append(build_layer(self.in_dim, self.hid_dim, self.gnn_type))
        for _ in range(self.num_layers - 1):
            self.convs.append(build_layer(self.hid_dim, self.hid_dim, self.gnn_type))

        if self.mode == 'node':
            self.cls = build_layer(self.hid_dim, self.num_classes, self.gnn_type)

        elif self.mode == 'graph':
            self.cls = nn.Linear(self.hid_dim, self.num_classes)

        if self.adv:
            self.domain_discriminator = nn.Linear(self.hid_dim, 2)
            
    def forward(self, data, prop_nums):

        if self.mode == 'node':
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.feat_bottleneck(x, edge_index, batch, prop_nums=prop_nums)
        x = self.feat_classifier(x, edge_index, batch, prop_nums=1)

        return x
    
    def feat_bottleneck(self, x, edge_index, batch, prop_nums=30):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, prop_nums=prop_nums)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.mode == 'graph':
            x = global_mean_pool(x, batch)

        return x
    
    def feat_classifier(self, x, edge_index, batch, prop_nums=1):

        if self.mode == 'node':
            x = self.cls(x, edge_index, prop_nums=prop_nums)
        else:
            x = self.cls(x)
        
        return x
    
    def domain_classifier(self, x, alpha):

        x = GradReverse.apply(x, alpha)
        d_logit = self.domain_discriminator(x)
        
        return d_logit
