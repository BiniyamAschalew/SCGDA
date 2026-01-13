import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


def build_layer(in_dim, out_dim, layer_type):
    layer_type = layer_type.lower()

    if layer_type == 'gcn':
        return GCNConv(in_dim, out_dim)

    elif layer_type == 'sage':
        return SAGEConv(in_dim, out_dim)

    elif layer_type == 'gat':
        return GATConv(in_dim, out_dim, heads=1, concat=False)

    elif layer_type == 'gin':
        return GINConv(Sequential(Linear(in_dim, out_dim)), train_eps=True)

    elif layer_type == 'prop':
        return PropGCNConv(in_dim, out_dim)
        
    elif layer_type == 'cgnn':
        return CGNNConv(in_dim, out_dim)

    else:
        raise ValueError(f'Invalid gnn backbone: {layer_type}')


def build_activation(name):
    act_name = name.lower()

    if act_name == 'relu':
        return torch.nn.ReLU()

    elif act_name == 'leaky_relu':
        return torch.nn.LeakyReLU()

    elif act_name == 'elu':
        return torch.nn.ELU()

    else:
        raise ValueError(f'Invalid activation function: {act_name}')