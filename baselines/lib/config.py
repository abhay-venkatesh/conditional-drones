from pathlib import Path
from shutil import copy2
import os
import yaml


def read_config_file(config_file_path):
    with open(config_file_path, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def build_paths(config_file_path):
    config = read_config_file(config_file_path)
    experiments_base_folder = Path("experiments")
    config["experiment folder"] = Path(experiments_base_folder, config["name"])
    config["stats folder"] = Path(config["experiment folder"], "stats")
    config["checkpoints folder"] = Path(config["experiment folder"],
                                        "checkpoints")

    for path in [
            experiments_base_folder, config["experiment folder"],
            config["stats folder"], config["checkpoints folder"]
    ]:
        if not os.path.exists(path):
            os.mkdir(path)

    copy2(config_file_path, config["experiment folder"])
    return config


def get_config(config_file_path):
    return build_paths(config_file_path)
