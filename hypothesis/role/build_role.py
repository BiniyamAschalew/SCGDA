from hypothesis.role.static import SignalRole
from hypothesis.role.static import GraphWave
from hypothesis.role.static import Struc2Vec


def build_role(config: dict):
    role_type = config["model"]["role_type"].lower()

    if role_type == "signal_role":
        steps = config["model"]["role_steps"]
        operator = config.get("operator", "sym_norm")
        return SignalRole(steps=steps, operator=operator)
    
    elif role_type == "graphwave":
        return GraphWave()
    
    elif role_type == "struc2vec":
        return Struc2Vec()
    
    else:
        raise ValueError(f"Unknown role type: {role_type}")