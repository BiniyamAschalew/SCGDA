from  models.baselines.gnn.gnn import GNN

from  models.ours.appnp.appnp_base import APPNPBase


class APPNPModel(GNN):
    """APPNP under the repo's standard GNN training wrapper."""

    def init_model(self, **kwargs):
        return APPNPBase(self.config).to(self.device)

