from models.baselines.gnn.gnn import GNN 
# from models.baselines.dane.dane import DANE
# from models.baselines.a2gnn.a2gn import A2GNN

from utils.expt_utils import print_string

def build_model(config: dict):
    
    model_name = config["model"]["name"]
    model = None

    if model_name.lower() == "gnn":
        model = GNN(config)

    # elif model_name.lower() == "dane":
    #     model = DANE(config)

    # elif model_name.lower() == "a2gnn":
    #     model = A2GNN(config)

    else:
        raise ValueError(f"Invalid model name {model_name}")

    print_string(f"=== {model_name} ===")
    return model