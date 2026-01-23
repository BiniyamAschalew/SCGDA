import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from models.__layers.ppmi_conv import PPMIConv
from torch_geometric.nn import global_mean_pool


class GNN(torch.nn.Module):
    """
    Notes
    -----
    - Supports both GCN and PPMI convolution types
    - Multiple layers with residual connections
    - Configurable activation and dropout
    """

    def __init__(self, in_dim, hid_dim, gnn_type='gcn', num_layers=3, act=F.relu, dropout=0.1, **kwargs):
        super(GNN, self).__init__()

        self.gnn_type = gnn_type
        self.act = act
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()

        if self.gnn_type == 'gcn':
            self.conv_layers.append(GCNConv(in_dim, hid_dim))

            for i in range(1, self.num_layers):
                self.conv_layers.append(GCNConv(hid_dim, hid_dim))
        else:
            self.conv_layers.append(PPMIConv(in_dim, hid_dim))

            for i in range(1, self.num_layers):
                self.conv_layers.append(PPMIConv(hid_dim, hid_dim))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, mode='node'):
        """
        Notes
        -----
        - Applies multiple GNN layers sequentially
        - Optional graph pooling for graph-level tasks
        - Dropout and activation between layers
        """
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
        
        if mode == 'graph':
            x = global_mean_pool(x, batch)
        
        return x


class AdaGCNBase(nn.Module):
    """
    Notes
    -----
    Architecture components:
    
    1. GNN encoder for feature extraction
    2. Classification layer
    3. Cross-entropy loss function
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 num_layers=3,
                 dropout=0.1,
                 act=F.relu,
                 gnn_type='gcn',
                 mode='node',
                 **kwargs):
        super(AdaGCNBase, self).__init__()

        self.encoder = GNN(in_dim=in_dim, hid_dim=hid_dim, gnn_type=gnn_type, act=act, num_layers=num_layers)
        self.cls_model = nn.Sequential(nn.Linear(hid_dim, num_classes))
        self.mode = mode
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, data):
        """
        Notes
        -----
        Process:

        1. Extract features based on mode (node/graph)
        2. Apply GNN encoder
        3. Return embeddings for downstream tasks
        """
        if self.mode == 'node':
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x, edge_index, batch, mode=self.mode)

        return x
