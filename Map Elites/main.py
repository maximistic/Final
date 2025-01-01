import os
import argparse
import yaml
from modules.evolution.evolution import evolve
from modules.utils.helpers import create_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    save_dir = config["save_dir"]
    create_dirs([save_dir])
    evolve(config)