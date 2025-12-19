from dataclasses import dataclass
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from plugins.loader import load_plugins
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os

log = logging.getLogger(__name__)

# Hydra configuration
@hydra.main(config_path="../configs", config_name="storm", version_base="1.3")  # Make sure the path and config_name match your setup
def main(cfg: DictConfig):
    seeds = cfg.splitter.seed  # List of seeds
    models = cfg.models  # List of models

    plugins = load_plugins(models) 

    for plugin in plugins:
        plugin.visualize_scatterplots(seeds)

if __name__ == "__main__":
    main()