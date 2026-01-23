from models.baselines.gnn.gnn import GNN 
from models.baselines.dane.dane import DANE
from models.baselines.simgda.simgda import SimGDA
from models.baselines.a2gnn.a2gnn import A2GNN
from models.baselines.grade.grade import GRADE
from models.baselines.strurw.strurw import StruRW
from models.baselines.dgsda.dgsda import DGSDA
from models.baselines.specreg.specreg import SpecReg
from models.baselines.mlp.mlp import MLP

from models.ours.simgda_role.simgda_role import SimGDARole
from models.ours.simgda_spectral.simgda_spectral import SimGDASpectral
from models.ours.structalign2.structalign2 import StructAlign2
from models.ours.simmlp.simmlp import SimMLP

from utils.expt_utils import print_string

def build_model(config: dict):
    
    model_name = config["model"]["name"].lower()
    model = None

    models_dict = {
        "gnn": GNN,
        "dane": DANE,
        "simgda": SimGDA,
        "grade": GRADE,
        "a2gnn": A2GNN,
        "strurw": StruRW,
        "dgsda": DGSDA,
        "specreg": SpecReg,
        "simgda_role": SimGDARole,
        "simgda_spectral": SimGDASpectral,
        "structalign2": StructAlign2,
        "mlp": MLP,
        "simmlp": SimMLP
    }

    if model_name in models_dict:
        model = models_dict[model_name](config)

    else:
        raise ValueError(f"Invalid model name {model_name}")

    print_string(f"=== {model_name} ===")
    return model
