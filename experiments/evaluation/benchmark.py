"""
Code for running deep evaluation benchmarks.
"""

import os
import time
import pandas as pd
import yaml

from  utils.config_utils import build_config
from  utils.expt_utils import to_valid_dir
from  run import run


config_dir = "./configs/expt_configs/bench1.yaml"
config = None
with open(config_dir, 'r') as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]

model_hps = config["model_hps"]
transfer_settings = config["transfer_settings"]
device = config["device"]
wandb = config["wandb"]

repeat = config["repeat"]
seed = config["seed"]
cur_time = time.strftime("%m%d_%H%M%S")

os.makedirs(output_dir, exist_ok=True)
result_dir = f"{output_dir}/benchmark_{cur_time}.csv"

combined_df = pd.DataFrame()

for r in range(repeat):
    for dataset, settings in transfer_settings.items():
        for source, target in settings.items():
            for model, hp in model_hps.items():

                config_setup = {
                    "data": dataset,
                    "expt": "default",
                    "model": model,
                }
                update_config = {
                    "expt":
                    {
                        "source": source,
                        "target": target,
                        "device": device,
                        "wandb_enabled": False,
                        "project": "SCGDA_Benchmark",
                        "seed": seed + r,

                    },
                    "model":{
                        "epochs": 200
                    }
                }

                config = build_config(config_setup, update_config,
                                    use_tuned=hp)
                result = run(config)

                result["source"] = source
                result["target"] = target
                result["seed"] = seed + r
                result["model"] = model
                result["dataset"] = dataset
                result["cur_time"] = time.strftime("%d%H%M%S")
                result["hp_setting"] = hp

                result_df = pd.DataFrame([result])
                combined_df = pd.concat([combined_df, result_df],
                                        ignore_index=True)

                combined_df.to_csv(result_dir, index=False)
                print(f"Intermediate results saved to {result_dir}")


print("Final Results:")
# print(combined_df)
print(f"All results saved to {result_dir}")

