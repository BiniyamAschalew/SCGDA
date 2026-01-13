import random
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import wandb
import pyfiglet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def print_string(string):
    ascii_art = pyfiglet.figlet_format(string)
    print(string)



class WandbHandler:
    """logging to wandb if it is enabled"""

    def __init__(self, config1: dict):

        config = deepcopy(config1)
        self.enabled = config["expt"]["wandb_enabled"]
        self.project = config["expt"]["project"]

        if config["model"]["name"].lower() == "a2gnn":
            config["model"]["name"] = "X2GDA"

        self.name = self._get_name(config)
        self.tag = config["expt"]["tag"]
        self.verbose = config["expt"]["verbose"]
        self.config = config
        s_t = f"{config['expt']['source']}->{config['expt']['target']}"

        self.config["s_t"] = s_t
        self.config["expt_id"] = config["expt"]["id"]

        self.run = None
        if self.enabled:
            self.run = wandb.init(
                project=self.project, name=self.name, tags=self.tag, config=self.config
            )

    def log(self, metrics: dict):
        if self.enabled:
            self.run.log(metrics)

        if self.verbose:
            print(metrics)

    def finish(self):
        if self.enabled:
            self.run.finish()

    def _get_name(self, config: dict):
        name = config["model"]["name"]

        s_t = ''
        gnn = config["model"]["gnn"]
        name = f"{name}_{s_t}_{gnn}"

        if config["expt"]["use_time"]:
            time = f"{datetime.now().strftime('%d|%H:%M')}"
            name = f"{name}_{time}"

        return name