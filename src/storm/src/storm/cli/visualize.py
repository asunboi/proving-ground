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

def main(cfg: DictConfig):
    seeds = cfg.splitter.seed  # List of seeds
    models = cfg.models  # List of models

    plugins = load_plugins(models) 

    for plugin in plugins:
        plugin.visualize_scatterplots(cfg)

if __name__ == "__main__":
    main()