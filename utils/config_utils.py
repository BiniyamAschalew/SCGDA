from omegaconf import OmegaConf
import os

# should be modified incase of different environments
CONFIG_DIR = "./configs"


def load_config(config_path: str):
    return OmegaConf.load(config_path)

def build_config(config_setup: dict, update_config: dict = None):
    configs = {}
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

    if cfg.expt.verbose:
        print(f"\nconfigs:\n\n  'expt': {cfg.expt}\n\n  'model': {cfg.model}\n\n  'data': {cfg.data}\n")

    return OmegaConf.to_container(cfg, resolve=True)
