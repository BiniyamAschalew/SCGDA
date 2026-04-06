from omegaconf import OmegaConf
import os

# should be modified incase of different environments
CONFIG_DIR = "./configs"
TUNED_DIR = "././__hps__/"

BASELINES = [os.path.splitext(f)[0] for f in os.listdir(f"{CONFIG_DIR}/model_configs/baselines") if f.endswith(".yaml")]
OURS = [os.path.splitext(f)[0] for f in os.listdir(f"{CONFIG_DIR}/model_configs/ours") if f.endswith(".yaml")]

def load_config(config_path: str):
    return OmegaConf.load(config_path)

def build_config(config_setup: dict, update_config: dict = None, borrow: str = None,
                 use_tuned: int = 0, use_default: bool = False) -> dict:
    configs = {}
    config_setup = dict(config_setup)

    raw_model = str(config_setup["model"]).lower()
    model = raw_model.split("/", 1)[-1]

    if raw_model.startswith("baselines/") and model in BASELINES:
        config_setup["model"] = f"baselines/{model}"
    elif raw_model.startswith("ours/") and model in OURS:
        config_setup["model"] = f"ours/{model}"
    elif model in BASELINES:
        config_setup["model"] = f"baselines/{model}"
    elif model in OURS:
        config_setup["model"] = f"ours/{model}"
    else:
        raise ValueError(f"Model '{raw_model}' not found in baselines or ours.")
    
    for config_type, config_name in config_setup.items():
        config_type = config_type.lower()
        config_name = config_name.lower()
        config_dir = f"{CONFIG_DIR}/{config_type}_configs/{config_name}.yaml"
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config file {config_dir} not found")
        configs[config_type] = load_config(config_dir)

    cfg = OmegaConf.create(configs)

    if update_config is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(update_config))

    if use_tuned:
        file = None
        if use_tuned == 1: # using the best settings from pygda benchmark
            file = "imported.yaml"
            TUNED_DIR = "././__hps__/imported"
        elif use_tuned == 2:
            file = "best.yaml"
            TUNED_DIR = "././__hps__/tuned"

        source = cfg.expt.source
        target = cfg.expt.target
        dataset = cfg.data.name.lower()

        if borrow:
            tuned_config = load_config(f"{TUNED_DIR}/{borrow}/{dataset}/{source}_{target}/{file}")
            tuned_config.name = model  # keep the original model name
        else:
            tuned_config = load_config(f"{TUNED_DIR}/{model}/{dataset}/{source}_{target}/{file}")
        
        cfg.model = OmegaConf.merge(cfg.model, tuned_config)
        
        # if cfg.expt.verbose == 2:
        print(f"\nLoaded tuned config for model '{model}' on pair ({source}, {target}):\n{tuned_config}\n")

    if cfg.expt.verbose:
        print(f"\nconfigs:\n\n  'expt': {cfg.expt}\n\n  'model': {cfg.model}\n\n  'data': {cfg.data}\n")

    return OmegaConf.to_container(cfg, resolve=True)
