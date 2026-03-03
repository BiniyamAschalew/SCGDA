
from models.base_model import BaseGDA
import torch


"""
For a given source-target pair, we learn separate Chebyshev operators 
on the source and target domains, with the goal of aligning the distributions of
the resulting node features after propagation
"""


class StructAlign(BaseGDA):
    def __init__(self, config: dict):
        super(StructAlign, self).__init__(config)

        self.config = config
        self.K = int(config["model"]["K"])
        self.source_params = torch.nn.Parameter(
            torch.ones(self.K)/self.K, requires_grad=True
        )