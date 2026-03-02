"""Let's create a model to take two datasets and 
aligns their edge indices (structure) to minimize their MMD for 
input probe distribution"""

import matplotlib.pyplot as plt
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from data.build_dataset import build_dataset
from utils.config_utils import build_config
from utils.expt_utils import set_seed


class Propagation(MessagePassing):
    """GCN-style normalized one-hop propagation."""

    def __init__(self, aggr: str = "add"):
        super().__init__(aggr=aggr)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

class ChebPropagation(MessagePassing):
    """No weight or feature transformation, just Chebyshev-style propagation."""
    
    def __init__(self, aggr: str = "add"):
        super().__init__(aggr=aggr)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, prop_param: torch.Tensor) -> torch.Tensor:
        """ Each param corresponds to a different chebyshev base, so we can do multiple propagations and combine them. """

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        norm_adj = torch.sparse.FloatTensor(edge_index, norm, torch.Size([x.size(0), x.size(0)]))
        
        # we compute the Chebyshev bases up to the order of prop_param, and then combine them with the corresponding parameters
        out = torch.zeros_like(x)
        T_k = x  # T_0(X) = X
        out += prop_param[0] * T_k
        if prop_param.size(0) > 1:
            T_k_1 = x  # T_{-1}(X) = X
            T_k = torch.sparse.mm(norm_adj, T_k)  # T_1(X) = A_hat X
            out += prop_param[1] * T_k
            for k in range(2, prop_param.size(0)):
                T_k_plus_1 = 2 * torch.sparse.mm(norm_adj, T_k) - T_k_1  # T_{k+1}(X) = 2 A_hat T_k(X) - T_{k-1}(X)
                out += prop_param[k] * T_k_plus_1
                T_k_1, T_k = T_k, T_k_plus_1

        return out
    

def train_aligner(src_data, tgt_data, config):

    src_params = torch.nn.Parameter(torch.randn(config["K"]))
    tgt_params = torch.nn.Parameter(torch.randn(config["K"]))
    cheb_prop = ChebPropagation()

    for epoch in range(config["epochs"]):

        src_out = cheb_prop(src_data.x, src_data.edge_index, src_params)
        tgt_out = cheb_prop(tgt_data.x, tgt_data.edge_index, tgt_params)

        # compute MMD loss between src_out and tgt_out
        mmd_loss = compute_mmd(src_out, tgt_out)
        smoothness_loss = compute_smoothness(src_out, src_data.edge_index) + compute_smoothness(tgt_out, tgt_data.edge_index)
        total_loss = mmd_loss + config["smoothness_weight"] * smoothness_loss

        # backpropagate and update parameters
        mmd_loss.backward()
        with torch.no_grad():
            src_params -= config["lr"] * src_params.grad
            tgt_params -= config["lr"] * tgt_params.grad
            src_params.grad.zero_()
            tgt_params.grad.zero_()

        if epoch % config["log_interval"] == 0:
            print(f"Epoch {epoch}, MMD Loss: {mmd_loss.item()}")

        




class Aligner(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
