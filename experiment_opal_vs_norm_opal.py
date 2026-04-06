import os
import time

import pandas as pd

from  run import run
from  utils.config_utils import build_config, load_config
from  utils.expt_utils import to_valid_dir


MODEL_SPECS = {
    0: {"name": "opal", "borrow": None},
    1: {"name": "norm_opal", "borrow": "opal"},
}

DATASETS = {
    0: "citation",
    1: "blog",
    2: "airport",
    3: "twitch",
    4: "mag",
}


REPEATS = 1
SEED = 2025
DEVICE = "cuda:2"

USE_TUNED = 2
USE_DEFAULT = False
WANDB = False
DIVERGENCE = "mmd"

id = {
    "model": [0, 1],
    "dataset": [2],
    "source": [0],
    "target": [1],
}

combined_df = pd.DataFrame()
result = {}
cur_time = time.strftime("%m%d%H%M%S")

for repeat in range(REPEATS):
    for mid in id["model"]:
        model_spec = MODEL_SPECS[mid]
        model = model_spec["name"]
        borrow_model = model_spec["borrow"]

        for did in id["dataset"]:
            dataset = DATASETS[did]

            data_config = load_config(f"./configs/data_configs/{dataset}.yaml")
            domains = data_config["domains"]

            src_ids = range(len(domains)) if len(id["source"]) == 0 else id["source"]
            tgt_ids = range(len(domains)) if len(id["target"]) == 0 else id["target"]

            for sid in src_ids:
                for tid in tgt_ids:
                    if sid == tid:
                        continue

                    if len(domains) <= max(sid, tid):
                        raise ValueError(
                            f"selected domain id {sid} or {tid} exceeds available domains: len={len(domains)}"
                        )

                    source = domains[sid]
                    target = domains[tid]

                    config_setup = {
                        "data": dataset,
                        "expt": "default",
                        "model": model,
                    }

                    update_config = {
                        "expt": {
                            "source": source,
                            "target": target,
                            "device": DEVICE,
                            "seed": SEED + repeat,
                            "wandb_enabled": WANDB,
                            "project": "SCGDA",
                        },
                        "model": {
                            "divergence": DIVERGENCE,
                        },
                    }

                    config = build_config(
                        config_setup,
                        update_config,
                        borrow=borrow_model,
                        use_tuned=USE_TUNED,
                        use_default=USE_DEFAULT,
                    )
                    result = run(config, from_pygda=False)

                    result["divergence"] = config["model"]["divergence"]
                    result["source"] = source
                    result["target"] = target
                    result["use_tuned"] = USE_TUNED
                    result["repeat"] = repeat
                    result["borrow_model"] = borrow_model or ""
                    result["tuned_from_model"] = borrow_model or model
                    result["model"] = model

                    result_df = pd.DataFrame([result])
                    combined_df = pd.concat([combined_df, result_df], ignore_index=True)


dataset_note = "_".join(DATASETS[did] for did in id["dataset"])
model_note = "_vs_".join(MODEL_SPECS[mid]["name"] for mid in id["model"])
notes = f"{dataset_note}_{model_note}"
result_dir = f"./__saved__/results/evaluation/{cur_time}_{notes}.csv"
os.makedirs(os.path.dirname(result_dir), exist_ok=True)
if os.path.exists(result_dir):
    result_dir = to_valid_dir(result_dir)

combined_df.to_csv(result_dir, index=False)
print(result)
print(f"Combined results saved to {result_dir}")
