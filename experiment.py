from utils.config_utils import build_config
from run import run

dataset = "citation"
source = "ACMv9"
target = "Citationv1"

# dataset = "blog"
# source = "Blog2"
# target = "Blog1"

# dataset = "airport"
# source = "BRAZIL"
# target = "EUROPE"

# dataset = "twitch"
# source = "DE"
# target = "EN"

# dataset = "mag"
# source = "MAG_FR"
# target = "MAG_JP"



config_setup = {
    "data": dataset,
    "expt": "default",
    "model": "a2gnn",
}

update_config = {
    "expt": {
        "source": source,
        "target": target,
        "device": "cuda:5",
        "seed": 42,
    }}

config = build_config(config_setup, update_config)

result = run(config)
print(result)