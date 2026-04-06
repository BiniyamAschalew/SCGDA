from torch import nn
import torch

from Learn.Clean_SCGDA.models.__filters.build_filter import build_filter


class VDABase(nn.Module):
    """two layer projector + """

    def __init__(self, config: dict):
        super(VDABase, self).__init__()

        self.in_dim = int(config["model"]["in_dim"])
        self.hid_dim = int(config["model"]["hid_dim"])
        self.num_classes = int(config["model"]["num_classes"])
        self.K = int(config["model"]["K"])
        self.dprate = float(config["model"]["dprate"])

        self.filter_name = str(config["model"]["filter_name"]).lower()
        self.filter_operator = build_filter(self.filter_name)()

        self.projector = nn.Sequential(
            nn.Dropout(p=self.dprate),
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dprate),
            nn.Linear(self.hid_dim, self.hid_dim),
        )

        self.cls = nn.Linear(self.hid_dim, self.num_classes)

    def reset_parameters(self):
        self.projector[1].reset_parameters()
        self.projector[4].reset_parameters()
        self.cls.reset_parameters()

    def aggregate_filter_stack(
        self,
        filter_stack: torch.Tensor,
        domain_param: torch.Tensor,
    ) -> torch.Tensor:
        """Per-feature weighted aggregation over basis dimension."""
        num_basis = self.K + 1
        if domain_param.shape != (num_basis, self.hid_dim):
            raise ValueError(
                f"domain_param shape mismatch: expected ({num_basis}, {self.hid_dim}), got {tuple(domain_param.shape)}"
            )
        return torch.einsum("ndk,kd->nd", filter_stack, domain_param)

    def filter_and_aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        domain_param: torch.Tensor,
        edge_weight=None,
    ):
        filter_stack = self.filter_operator.return_stack(
            h,
            edge_index,
            self.K + 1,
            edge_weight=edge_weight,
        )
        filtered = self.aggregate_filter_stack(filter_stack, domain_param)
        return filter_stack, filtered


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        domain_param: torch.Tensor,
        edge_weight=None,
    ) -> torch.Tensor:
        h = self.projector(x)
        _, filtered = self.filter_and_aggregate(h, edge_index, domain_param, edge_weight=edge_weight)
        return self.cls(filtered)
