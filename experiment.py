import time
import os

from utils.config_utils import build_config, load_config
from utils.expt_utils import to_valid_dir
from run import run
import pandas as pd


BASELINES = {"gnn", "dane", "simgda", 
             "grade", "a2gnn", "strurw", 
             "dgsda", "specreg"}
MODELS = {
    0:"gnn",   1: "dane", 
    2:"simgda",   3:"grade", 
    4:"a2gnn",   5:"strurw", 
    6:"dgsda",   7:"specreg",
    8:"simgda_role", 9:"simgda_spectral",
    10:"structalign2", 11:"mlp", 12:"simmlp",
    13:"acdne", 14:"asn", 15:"adagcn",
    16:"dlit", 17:"simgda_cheb",
    }


DATASETS = {
    0:"citation", 1:"blog", 
    2:"airport", 3:"twitch", 4:"mag"
    }


ROLE_TYPES = {
    0:"random_role", 1:"graphwave", 
    2:"signal_role"
    }

REPEATS = 2
role_type = ROLE_TYPES[2]

SEED = 0
# EPOCHS = 200
DEVICE = "cuda:1"

USE_TUNED = 0

BORROW = "simgda"
# BORROW = None

USE_DEFAULT = False
WANDB = False
FROM_PYGDA = False

id = {
    "model": [17],
    "dataset": [1, 2],
    "source": [0],
    "target": [1],
}
notes = "dlit_evaluation"

combined_df = pd.DataFrame()
cur_time = time.strftime("%d%H%M%S")

for repeat in range(REPEATS):
    for mid in id["model"]:
        for did in id["dataset"]:

            model = MODELS[mid][:]

            dataset = DATASETS[did]

            data_config = load_config(f"./configs/data_configs/{dataset}.yaml")
            domains = data_config["domains"]

            src_ids = range(len(domains)) if len(id["source"]) == 0 else id["source"]
            tgt_ids = range(len(domains)) if len(id["target"]) == 0 else id["target"]

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
                            "seed": SEED + repeat,
                            # "epochs": EPOCHS,
                            "wandb_enabled": WANDB,
                            "project": "SCGDA",
                        },
                        "model":
                        {
                            "adv": False,
                            "role_type": role_type,
                        },
                        
                    }

                    config = build_config(config_setup, update_config, borrow=BORROW,
                                          use_tuned=USE_TUNED, use_default=USE_DEFAULT)
                    result = run(config, from_pygda=FROM_PYGDA)

                    result["repeat"] = repeat
                    result["source"] = source
                    result["target"] = target
                    result["cur_time"] = time.strftime("%d%H%M")
                    result["use_tuned"] = USE_TUNED

                    model_name = model[:]
                    if "role" in model_name:
                        model_name += f"_{config['model']['role_type']}"

                    result["model"] = model_name


                    result_df = pd.DataFrame([result])
                    combined_df = pd.concat([combined_df, result_df], 
                                            ignore_index=True)



result_dir = f"./__saved__/results/evaluation/{cur_time}_{notes}.csv"
if os.path.exists(result_dir):
    result_dir = to_valid_dir(result_dir)

combined_df.to_csv(result_dir, index=False)
print(result)
print(f"Combined results saved to {result_dir}")
