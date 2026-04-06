"""In this experiment we evaluate the perforance degradation of a GNN
with respect to distribution shift """

from  models.baselines.gnn.gnn import GNN
from  models.baselines.simgda.simgda import SimGDA


if __name__ == "__main__":


    config = {
        "model": {
            "name": "gnn",
            "batch_size": 32,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "use_mask": True
        },
        "expt": {
            "epochs": 200,
            "verbose": True
        }
    }

    gnn = GNN(config)
    simgda = SimGDA(config)