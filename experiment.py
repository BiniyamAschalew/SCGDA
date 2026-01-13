from utils.config_utils import build_config
from run import run

dataset = "citation"
source = "ACMv9"
target = "DBLPv7"

dataset = "blog"
source = "Blog2"
target = "Blog1"

dataset = "airport"
source = "USA"
target = "Brazil"

dataset = "twitch"
source = "DE"
target = "EN"

dataset = "mag"
source = "MAG_FR"
target = "MAG_JP"



config_setup = {
    "data": dataset,
    "expt": "default",
    "model": "gnn",
}

update_config = {
    "expt": {
        "source": source,
        "target": target,
        "device": "cuda:0",
        "seed": 42,
    }}

config = build_config(config_setup, update_config)

result = run(config)
print(result)