import torch
from torch import nn
import torch.nn.functional as F
from models.layers.build_layer import build_layer, build_activation

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GRADEBase(nn.Module):


    def __init__(self, config: dict):
        super(GRADEBase, self).__init__()
        
        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]

        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        self.act = build_activation(config["model"]["activation"])

        self.mode = config["model"]["mode"]
        disc = config["model"]["disc"]

        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(self.in_dim, self.hid_dim))

        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hid_dim, self.hid_dim))

        self.cls = nn.Linear(self.hid_dim, self.num_classes)

        if disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(
                    self.hid_dim * self.num_layers + self.num_classes, 2)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(
                    self.hid_dim * self.num_layers + self.num_classes * 2, 2)
            )

        self.criterion = nn.CrossEntropyLoss()
            
    def forward(self, data):

        if self.mode == 'node':
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        x, feat_list = self.feat_bottleneck(x, edge_index, batch)
        x = self.feat_classifier(x)

        feat_list.append(x)
        feat_list = torch.cat(feat_list, dim=1)

        return x, feat_list
    
    def feat_bottleneck(self, x, edge_index, batch):

        feat_list = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.mode == 'node':
                feat_list.append(x)
            else:
                feat_list.append(global_mean_pool(x, batch))
        
        if self.mode == 'graph':
            x = global_mean_pool(x, batch)

        return x, feat_list
    
    def feat_classifier(self, x):
        x = self.cls(x)
        return x
    
    def one_hot_embedding(self, labels):
        y = torch.eye(self.num_classes, device=labels.device)
        return y[labels]
