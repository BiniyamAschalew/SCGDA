from  hypothesis.role.static.signal_role import SignalRole
from  hypothesis.role.static.graphwave import GraphWave
from  hypothesis.role.static.struc2vec import Struc2Vec
from  hypothesis.role.static.random_role import RandomRole


def build_role(config: dict):
    role_type = config["model"]["role_type"].lower()

    if role_type == "signal_role":
        return SignalRole(config)
    
    elif role_type == "graphwave":
        return GraphWave(config)
    
    elif role_type == "struc2vec":
        return Struc2Vec(config)
    
    elif role_type == "random_role":
        return RandomRole(config)
    
    else:
        raise ValueError(f"Unknown role type: {role_type}")