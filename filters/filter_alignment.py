import torch
from torch import nn
import torch_geometric
# import lightGCN prop
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


