import yaml
import os
from copy import deepcopy


# should be modified incase of different environments
CONFIG_DIR = "./configs"


def load_config(config_path: str):
    """simply, load a yaml config file"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def do_update_config(configs: dict, update_config: dict):
    """update the config with the update_config"""
    for key, value in update_config.items():

        if isinstance(value, dict):
            if key in configs:
                configs[key] = do_update_config(configs[key], value)
            else:
                configs[key] = value
        else:
            configs[key] = value

    return configs


def build_config(config_setup: dict, update_config: dict = None):
    """create a config file for the experiment"""

    configs = {}

    for config_type, config_name in config_setup.items():

        config_type = config_type.lower()
        config_name = config_name.lower()

        config_dir = f"{CONFIG_DIR}/{config_type}_configs/{config_name}.yaml"

        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config file {config_dir} not found")

        sub_config = load_config(config_dir)
        configs[config_type] = sub_config

    if update_config is not None:
        configs = do_update_config(configs, update_config)

    if configs["expt"]["verbose"]:
        print(f"\nconfigs: \n\n\n  'expt': {configs['expt']}\n\n\n  'model': {configs['model']}\n\n\n  'data': {configs['data']}\n")

    return configs