from utils.config_utils import build_config
from run import run


config_setup = {
    "data": "citation",
    "expt": "default",
    "model": "gnn",
}

config = build_config(config_setup)

result = run(config)
print(result)