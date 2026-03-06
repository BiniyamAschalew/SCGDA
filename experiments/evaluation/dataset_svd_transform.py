import time
import os

from utils.config_utils import build_config, load_config
from utils.expt_utils import to_valid_dir
from run import run
import pandas as pd


SEED = 0
EPOCHS = 200
DEVICE = "cuda:6"
MODELS = {
    0:"gnn", 1: "dane", 2:"simgda", 
    3:"grade", 4:"a2gnn", 5:"strurw", 
    6:"dgsda", 7:"specreg"}
DATASETS = ["citation", "blog", "airport", "twitch", "mag"]

id = {
    "model": [0],
    "dataset": [0],
    "source": [0],
    "target": [1],
}
notes = "check_model"

combined_df = pd.DataFrame()
cur_time = time.strftime("%d%H%M")

for mid in id["model"]:
    for dataset in DATASETS[::-1]:

        model = MODELS[mid]

        config_setup = {
            "data": dataset,
            "expt": "default",
            "model": model,
        }

        data_config = load_config(f"./configs/data_configs/{dataset}.yaml")
        domains = data_config["domains"]

        for source in domains:
            for target in domains:
                
                # exclude same source and target
                if source == target:
                    continue

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
