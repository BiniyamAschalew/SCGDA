import time
import os

from utils.config_utils import build_config, load_config
from utils.expt_utils import to_valid_dir
from run import run
import pandas as pd

SEED = 0
EPOCHS = 200
DEVICE = "cuda:7"
REPEATS = 3

BASELINES = {"gnn", "dane", "simgda", 
             "grade", "a2gnn", "strurw", 
             "dgsda", "specreg"}
MODELS = {
    0:"gnn", 1: "dane", 
    2:"simgda", 3:"grade", 
    4:"a2gnn", 5:"strurw", 
    6:"dgsda", 7:"specreg"}

DATASETS = {0:"citation", 1:"blog", 
            2:"airport", 3:"twitch", 4:"mag"}

id = {
    "model": [7],
    "dataset": [0],
    "source": [],
    "target": [],
}
notes = "process_datasets"

combined_df = pd.DataFrame()
cur_time = time.strftime("%d%H%M")

for repeat in range(REPEATS):
    for mid in id["model"]:
        for did in id["dataset"]:

            model = MODELS[mid]
            # accounding for the directory structure
            if model in BASELINES:
                model = "baselines/" + model
            else:
                model = "ours/" + model

            dataset = DATASETS[did]

            data_config = load_config(f"./configs/data_configs/{dataset}.yaml")
            domains = data_config["domains"]

            src_ids = domains if len(id["source"]) == 0 else id["source"]
            tgt_ids = domains if len(id["target"]) == 0 else id["target"]

            for sid in src_ids:
                for tid in tgt_ids:
                    
                    # exclude same source and target
                    if sid == tid:
                        continue

                    if len(domains) <= max(sid, tid):
                        raise ValueError(f"selected domain id {sid} or {tid} exceeds available domains: len={len(domains)}")

                    source = domains[sid]
                    target = domains[tid]

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
                            "device": DEVICE,
                            "seed": SEED,
                            "epochs": EPOCHS,
                        },
                        "model":
                        {
                            "adv": False,
                        },
                        
                    }

                    config = build_config(config_setup, update_config)
                    result = run(config)

                    result_df = pd.DataFrame([result])
                    combined_df = pd.concat([combined_df, result_df], 
                                            ignore_index=True)



result_dir = f"./__saved__/results/evaluation/{cur_time}_{notes}.csv"
if os.path.exists(result_dir):
    result_dir = to_valid_dir(result_dir)

result_df.to_csv(result_dir, index=False)
print(result)
print(f"Combined results saved to {result_dir}")
