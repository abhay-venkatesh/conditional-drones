from lib.config import get_config
from pathlib import Path
import argparse
import importlib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("mode", help="Options: [train eval_train test]")
    parser.add_argument(
        "--from_checkpoint",
        help="whether to continue training from checkpoint",
        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = get_config(Path(args.config_file))
    if args.mode == "train":
        Trainer = importlib.import_module("lib.train").Trainer
        Trainer(config).train()
    elif args.mode == "eval_train":
        raise NotImplementedError
