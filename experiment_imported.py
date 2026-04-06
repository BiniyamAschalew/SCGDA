import time
import os

from  utils.config_utils import build_config, load_config
from  utils.expt_utils import to_valid_dir
from  run import run
import pandas as pd


BASELINES = {"gnn", "dane", "simgda", 
             "grade", "a2gnn", "strurw", 
             "dgsda", "specreg", "kbl", "pairalign"}


# MODELS = {0:"A2GNN", 1:"ADAGCN", 2:"DANE", 3:"DGSDA",
#           4:"GRADE", 5:"JHGDA", 6:"KBL",
#           7:"PAIRALIGN", 8:"SPECREG", 9:"STRURW",
#           10:"TDSS", 11:"UDAGCN",
#           }

MODELS = [
    "KBL", "DANE", "GRADE", "A2GNN", "TDSS",
    "A2GNN", "ADAGCN", "DGSDA", "SPECREG", "KBL", "PAIRALIGN",
    "JHGDA","STRURW", ]


TRANSFER_SCENARIOS = {
    "airport": [
        ("USA", "EUROPE"), ("USA", "BRAZIL"),
        # ("EUROPE", "USA"), ("EUROPE", "BRAZIL"),
        # ("BRAZIL", "USA"), ("BRAZIL", "EUROPE"),
    ],
    "blog": [
        ("Blog1", "Blog2"), ("Blog2", "Blog1"),
    ],
    "citation": [
        ("Citationv1", "ACMv9"), ("Citationv1", "DBLPv7"),
        # ("DBLPv7", "ACMv9"), ("DBLPv7", "Citationv1"),
        # ("ACMv9", "Citationv1"), ("ACMv9", "DBLPv7"),
    ],
    "twitch": [
        ("EN", "DE"), ("DE", "EN"),
    ],
}


# DATASETS = {
#     0:"citation", 1:"blog", 
#     2:"airport", 3:"twitch", 4:"mag"
#     }


# ROLE_TYPES = {
#     0:"random_role", 1:"graphwave", 
#     2:"signal_role"
#     }

REPEATS = 1
# role_type = ROLE_TYPES[2]

SEED = 0
EPOCHS = 3
DEVICE = "cuda:6"

USE_TUNED = 1

BORROW = None
# BORROW = "dgsda"
# BORROW = "adagcn"

USE_DEFAULT = False
WANDB = False
FROM_PYGDA = True

# id = {
#     "model": [0,1,2,3,4,5,6,7,8,9,10,11],
#     "dataset": [0,1],
#     "source": [0],
#     "target": [1],
# }

combined_df = pd.DataFrame()
cur_time = time.strftime("%m%d%H%M%S")

for repeat in range(REPEATS):
    for model in MODELS:
        for dataset in TRANSFER_SCENARIOS:
            for source, target in TRANSFER_SCENARIOS[dataset]:

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
                        "model":{
                            "epochs": EPOCHS,
                        }
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


notes = f"{dataset}_{model}"
result_dir = f"./__saved__/results/evaluation/{cur_time}_{notes}.csv"
os.makedirs(os.path.dirname(result_dir), exist_ok=True)
if os.path.exists(result_dir):
    result_dir = to_valid_dir(result_dir)

combined_df.to_csv(result_dir, index=False)
print(result)
print(f"Combined results saved to {result_dir}")
