from models.baselines.gnn.gnn import GNN 
from models.baselines.dane.dane import DANE
from models.baselines.simgda.simgda import SimGDA
from models.baselines.a2gnn.a2gnn import A2GNN
from models.baselines.grade.grade import GRADE
from models.baselines.strurw.strurw import StruRW
from models.baselines.dgsda.dgsda import DGSDA
from models.baselines.specreg.specreg import SpecReg
from models.baselines.mlp.mlp import MLP
from models.baselines.acdne.acdne import ACDNE
from models.baselines.asn.asn import ASN
from models.baselines.adagcn.adagcn import AdaGCN

# from models.ours.simgda_role.simgda_role import SimGDARole
# from models.ours.simgda_cheb.simgda_cheb import SimGDACheb
from models.ours.simgda_spectral.simgda_spectral import SimGDASpectral
from models.ours.structalign2.structalign2 import StructAlign2
from models.ours.simmlp.simmlp import SimMLP
from models.ours.dlite.dlite import DLIT
from models.ours.bdlite.bdlite import BDlite
from models.ours.scgda.scgda import SCGDA
from models.ours.test.test import Test
from models.ours.fda.fda import FDA
from models.ours.simgda_filter.simgda_filter import SimGDAFilter
from models.ours.filtada.filtada import FiltADA
from models.ours.adaf.adaf import ADAF
from models.ours.adgfn.adgfn import ADGFN
from models.ours.dgf.dgf import DGF
from models.ours.fan.fan import FAN

# Imported models from pygda
from models.imported.a2gnn.a2gnn import A2GNN as A2GNN_imported
from models.imported.adagcn.adagcn import AdaGCN as AdaGCN_imported
from models.imported.dgsda.dgsda import DGSDA as DGSDA_imported
from models.imported.specreg.specreg import SpecReg as SpecReg_imported
from models.imported.kbl.kbl import KBL as KBL_imported
from models.imported.pairalign.pairalign import PairAlign as PairAlign_imported


from utils.expt_utils import print_string

def build_model(config: dict, from_pygda: bool = False):
    
    model_name = config["model"]["name"].lower()
    model = None

    imported_models_dict = {
        "a2gnn": A2GNN_imported,
        "adagcn": AdaGCN_imported,
        "dgsda": DGSDA_imported,
        "specreg": SpecReg_imported,
        "kbl": KBL_imported,
        "pairalign": PairAlign_imported,
    }

    models_dict = {
        "gnn": GNN,
        "dane": DANE,
        "simgda": SimGDA,
        "grade": GRADE,
        "a2gnn": A2GNN,
        "strurw": StruRW,
        "dgsda": DGSDA,
        "specreg": SpecReg,
        "simgda_spectral": SimGDASpectral,
        "structalign2": StructAlign2,
        "mlp": MLP,
        "simmlp": SimMLP,
        "acdne": ACDNE,
        "asn": ASN,
        "adagcn": AdaGCN,
        "dlit": DLIT,
        "bdlite": BDlite,
        "scgda": SCGDA,
        "fda": FDA,
        "simgda_filter": SimGDAFilter,
        "filtada": FiltADA,
        "adaf": ADAF,
        "adgfn": ADGFN,
        "fan": FAN,
        # "simgda_role": SimGDARole,
        # "simgda_cheb": SimGDACheb,
        "test": Test,
        "dgf": DGF,
    }

    if from_pygda:
        if model_name in imported_models_dict:
            model = imported_models_dict[model_name](config)
        else:
            raise ValueError(f"Invalid model name {model_name} for imported models")
    else:   
        if model_name in models_dict:
            model = models_dict[model_name](config)

        else:
            raise ValueError(f"Invalid model name {model_name}")

    print_string(f"=== {model_name} ===")
    return model
