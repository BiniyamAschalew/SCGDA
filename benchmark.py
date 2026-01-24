"""
Code for running deep evaluation benchmarks.
"""

import os
import time
import pandas as pd
import yaml
import argparse

from utils.config_utils import build_config
from run import run


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=200,
                    help='random seed for the benchmark run')
parser.add_argument("--config", type=str, default="bench1",)

args = parser.parse_args()
seed = args.seed
config_file = args.config

config_dir = f"./configs/expt_configs/{config_file}.yaml"
config = None
with open(config_dir, 'r') as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

model_hps = config["model_hps"]
transfer_settings = config["transfer_settings"]
device = config["device"]
wandb = config["wandb"]
verbose = config.get("verbose", 0)

prefix = config.get("prefix", None)
cur_time = time.strftime("%m%d_%H%M%S")
note = config["bench_note"]
result_dir = f"{output_dir}/benchmark_{note}_{cur_time}.csv"

combined_df = pd.DataFrame()

# for r in range(repeat):
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
                    "seed": seed,
                    "verbose": verbose,

                },
                "model":{
                    "epochs": 200
                }
            }

            run_config = build_config(config_setup, update_config,
                                use_tuned=hp)
            result = run(run_config)

            result["source"] = source
            result["target"] = target
            result["seed"] = seed

            if prefix is not None:
                model = prefix + model

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
print(f"All results saved to {result_dir}")

