import time
import os

from utils.config_utils import build_config, load_config
from utils.expt_utils import to_valid_dir
from run import run
import pandas as pd




SEED = 0
EPOCHS = 200
DEVICE = "cuda:7"
MODELS = ["gnn", "dane", "simgda", "a2gnn"]
DATASETS = ["citation", "blog", "airport", "twitch", "mag"]

id = {
    "model": [3],
    "dataset": [0],
    "source": [0],
    "target": [1],
}
notes = "a2gnn_adv_test"

combined_df = pd.DataFrame()
cur_time = time.strftime("%d%H%M")

for mid in id["model"]:
    for did in id["dataset"]:
        for sid in id["source"]:
            for tid in id["target"]:
                
                # exclude same source and target
                if sid == tid:
                    continue


                model = MODELS[mid]
                dataset = DATASETS[did]

                data_config = load_config(f"./configs/data_configs/{dataset}.yaml")
                domains = data_config["domains"]

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
