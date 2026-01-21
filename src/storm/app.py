from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig

_CMD_ENV = "STORM_CMD"


def _dispatch(cmd: str, cfg: DictConfig) -> None:
    if cmd == "run":
        from storm.cli.run import main as fn
    elif cmd == "init":
        from storm.cli.init import main as fn
    elif cmd == "visualize":
        from storm.cli.visualize import main as fn
    else:
        raise ValueError(f"Unknown command: {cmd!r} (expected run/init/visualize)")
    fn(cfg)


@hydra.main(config_path="../configs", config_name="storm", version_base="1.3")
def hydra_main(cfg: DictConfig) -> None:
    cmd = os.environ.get(_CMD_ENV, "run")
    _dispatch(cmd, cfg)