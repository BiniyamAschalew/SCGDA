from Learn.Clean_SCGDA.models.baselines.gnn.gnn import GNN

from Learn.Clean_SCGDA.models.ours.gprgnn.gprgnn_base import GPRGNNBase


class GPRGNNModel(GNN):
    """GPR-GNN under the repo's standard GNN training wrapper."""

    def init_model(self, **kwargs):
        return GPRGNNBase(self.config).to(self.device)

