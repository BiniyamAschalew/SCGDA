from Learn.Clean_SCGDA.models.baselines.gnn.gnn import GNN 
from Learn.Clean_SCGDA.models.baselines.dane.dane import DANE
from Learn.Clean_SCGDA.models.baselines.simgda.simgda import SimGDA
from Learn.Clean_SCGDA.models.baselines.a2gnn.a2gnn import A2GNN
from Learn.Clean_SCGDA.models.baselines.grade.grade import GRADE
from Learn.Clean_SCGDA.models.baselines.strurw.strurw import StruRW
from Learn.Clean_SCGDA.models.baselines.dgsda.dgsda import DGSDA
from Learn.Clean_SCGDA.models.baselines.specreg.specreg import SpecReg
from Learn.Clean_SCGDA.models.baselines.mlp.mlp import MLP
from Learn.Clean_SCGDA.models.baselines.acdne.acdne import ACDNE
from Learn.Clean_SCGDA.models.baselines.asn.asn import ASN
from Learn.Clean_SCGDA.models.baselines.adagcn.adagcn import AdaGCN
from Learn.Clean_SCGDA.models.baselines.gnn.source_only_gnn import SourceOnlyGNN
from Learn.Clean_SCGDA.models.baselines.simgda.tracked_simgda import TrackedSimGDA

# from models.ours.simgda_role.simgda_role import SimGDARole
# from models.ours.simgda_cheb.simgda_cheb import SimGDACheb
from Learn.Clean_SCGDA.models.ours.simgda_spectral.simgda_spectral import SimGDASpectral
from Learn.Clean_SCGDA.models.ours.structalign2.structalign2 import StructAlign2
from Learn.Clean_SCGDA.models.ours.simmlp.simmlp import SimMLP
from Learn.Clean_SCGDA.models.ours.dlite.dlite import DLIT
from Learn.Clean_SCGDA.models.ours.bdlite.bdlite import BDlite
from Learn.Clean_SCGDA.models.ours.scgda.scgda import SCGDA
from Learn.Clean_SCGDA.models.ours.test.test import Test
from Learn.Clean_SCGDA.models.ours.fda.fda import FDA
from Learn.Clean_SCGDA.models.ours.simgda_filter.simgda_filter import SimGDAFilter
from Learn.Clean_SCGDA.models.ours.filtada.filtada import FiltADA
from Learn.Clean_SCGDA.models.ours.adaf.adaf import ADAF
from Learn.Clean_SCGDA.models.ours.adgfn.adgfn import ADGFN
from Learn.Clean_SCGDA.models.ours.dgf.dgf import DGF
from Learn.Clean_SCGDA.models.ours.fan.fan import FAN
from Learn.Clean_SCGDA.models.ours.opal.opal import OPAL
from Learn.Clean_SCGDA.models.ours.norm_opal.norm_opal import NormOPAL
from Learn.Clean_SCGDA.models.ours.appnp.appnp import APPNPModel
from Learn.Clean_SCGDA.models.ours.gprgnn.gprgnn import GPRGNNModel

# Imported models from pygda
from Learn.Clean_SCGDA.models.imported.a2gnn.a2gnn import A2GNN as A2GNN_imported
from Learn.Clean_SCGDA.models.imported.adagcn.adagcn import AdaGCN as AdaGCN_imported
from Learn.Clean_SCGDA.models.imported.dane.dane import DANE as DANE_imported
from Learn.Clean_SCGDA.models.imported.dgsda.dgsda import DGSDA as DGSDA_imported
from Learn.Clean_SCGDA.models.imported.grade.grade import GRADE as GRADE_imported
from Learn.Clean_SCGDA.models.imported.jhgda.jhgda import JHGDA as JHGDA_imported
from Learn.Clean_SCGDA.models.imported.kbl.kbl import KBL as KBL_imported
from Learn.Clean_SCGDA.models.imported.pairalign.pairalign import PairAlign as PairAlign_imported
from Learn.Clean_SCGDA.models.imported.specreg.specreg import SpecReg as SpecReg_imported
from Learn.Clean_SCGDA.models.imported.strurw.strurw import StruRW as StruRW_imported
from Learn.Clean_SCGDA.models.imported.tdss.tdss import TDSS as TDSS_imported
from Learn.Clean_SCGDA.models.imported.udagcn.udagcn import UDAGCN as UDAGCN_imported

from Learn.Clean_SCGDA.utils.expt_utils import print_string

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
        "dane": DANE_imported,
        "grade": GRADE_imported,
        "strurw": StruRW_imported,
        "jhgda": JHGDA_imported,
        "tdss": TDSS_imported,
        "udagcn": UDAGCN_imported,
    }

    models_dict = {
        "gnn": GNN,
        "source_only_gnn": SourceOnlyGNN,
        "sourceonlygnn": SourceOnlyGNN,
        "dane": DANE,
        "simgda": SimGDA,
        "tracked_simgda": TrackedSimGDA,
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
        "opal": OPAL,
        "norm_opal": NormOPAL,
        "appnp": APPNPModel,
        "gprgnn": GPRGNNModel,
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
