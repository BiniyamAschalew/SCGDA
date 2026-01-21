from pygda.models import UDAGCN, A2GNN, GRADE
from pygda.models import ASN, SpecReg, GNN
from pygda.models import StruRW, ACDNE, DANE
from pygda.models import AdaGCN, JHGDA, KBL
from pygda.models import DGDA, SAGDA, CWGCN
from pygda.models import DMGNN, PairAlign, DGSDA

def build_pygda_model(config: dict):
    
    model_name = config["model"]["name"]
    in_dim = config["model"]["in_dim"]
    hid_dim = config["model"]["hid_dim"]
    out_dim = config["model"]["num_classes"]
    device = config["expt"]["device"]

    if model_name.lower() == "udagcn":
        model = UDAGCN(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "a2gnn":
        model = A2GNN(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "grade":
        model = GRADE(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "asn":
        hid_dim_vae = config["model"].get("hid_dim_vae", hid_dim)
        model = ASN(in_dim=in_dim, hid_dim=hid_dim, hid_dim_vae=hid_dim_vae, num_classes=out_dim, device=device)
    elif model_name.lower() == "specreg":
        model = SpecReg(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device, reg_mode=True)
    elif model_name.lower() == "gnn":
        model = GNN(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "strurw":
        model = StruRW(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "acdne":
        model = ACDNE(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "dane":
        model = DANE(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "adagcn":
        model = AdaGCN(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "jhgda":
        model = JHGDA(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "kbl":
        model = KBL(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "dgda":
        model = DGDA(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "sagda":
        model = SAGDA(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "cwgcn":
        model = CWGCN(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "dmgnn":
        model = DMGNN(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "pairalign":
        model = PairAlign(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    elif model_name.lower() == "dgsda":
        model = DGSDA(in_dim=in_dim, hid_dim=hid_dim, num_classes=out_dim, device=device)
    else:
        raise ValueError(f"Invalid model name {model_name}")

    print(f"\n=== Loaded {model_name} from pygda module ===\n")
    return model

