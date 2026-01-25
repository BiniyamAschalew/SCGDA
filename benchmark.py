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
parser.add_argument("--run_dir", type=str, default=None,
                    help='directory to save results (created by run_benchmark.sh)')

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
note = config["bench_note"]

# Use run_dir if provided (from run_benchmark.sh), otherwise create timestamped dir
if args.run_dir:
    save_dir = args.run_dir
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
else:
    cur_time = time.strftime("%m%d_%H%M%S")
    save_dir = f"{output_dir}/run_{cur_time}"
    os.makedirs(save_dir, exist_ok=True)

result_dir = f"{save_dir}/benchmark_{note}_seed{seed}.csv"

print(f"[Seed {seed}] Starting benchmark, will save to: {result_dir}")

combined_df = pd.DataFrame()

# Calculate total experiments for progress tracking
total_experiments = sum(
    len(settings) * len(model_hps) 
    for settings in transfer_settings.values()
)
current_exp = 0

# for r in range(repeat):
for dataset, settings in transfer_settings.items():
    for source, target in settings.items():
        for model, hp in model_hps.items():
            current_exp += 1
            print(f"[Seed {seed}] Running experiment {current_exp}/{total_experiments}: {model} on {dataset} ({source}->{target})")

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
            
            try:
                result = run(run_config)
                result["status"] = "ok"
            except Exception as e:
                print(f"[Seed {seed}] ERROR in {model} on {dataset} ({source}->{target}): {e}")
                result = {
                    "micro_f1": None,
                    "macro_f1": None,
                    "status": f"error: {str(e)[:50]}",
                    "train_time": None,
                }

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
            print(f"[Seed {seed}] Saved intermediate results ({current_exp}/{total_experiments}) to {result_dir}")


print(f"[Seed {seed}] Benchmark complete! All {total_experiments} results saved to {result_dir}")

