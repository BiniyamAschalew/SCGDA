from models.baselines.gnn.gnn import GNN 
from models.baselines.dane.dane import DANE
from models.baselines.simgda.simgda import SimGDA
from models.baselines.a2gnn.a2gnn import A2GNN
from models.baselines.grade.grade import GRADE
from models.baselines.strurw.strurw import StruRW
from models.baselines.dgsda.dgsda import DGSDA
from models.baselines.specreg.specreg import SpecReg

from models.ours.simgda_role.simgda_role import SimGDARole
from models.ours.simgda_spectral.simgda_spectral import SimGDASpectral

from utils.expt_utils import print_string

def build_model(config: dict):
    
    model_name = config["model"]["name"]
    model = None

    if model_name.lower() == "gnn":
        model = GNN(config)

    elif model_name.lower() == "dane":
        model = DANE(config)

    elif model_name.lower() == "simgda":
        model = SimGDA(config)

    elif model_name.lower() == "grade":
        model = GRADE(config)

    elif model_name.lower() == "a2gnn":
        model = A2GNN(config)

    elif model_name.lower() == "strurw":
        model = StruRW(config)

    elif model_name.lower() == "dgsda":
        model = DGSDA(config)

    elif model_name.lower() == "specreg":
        model = SpecReg(config)

    elif model_name.lower() == "simgda_role":
        model = SimGDARole(config)

    elif model_name.lower() == "simgda_spectral":
        model = SimGDASpectral(config)

    else:
        raise ValueError(f"Invalid model name {model_name}")

    print_string(f"=== {model_name} ===")
    return model