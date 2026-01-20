# import argparse as ap

# from hydra import compose, initialize
# from omegaconf import DictConfig

# from ._cli import (
#     add_arguments_emb,
#     add_arguments_tx,
#     run_emb_fit,
#     run_emb_transform,
#     run_emb_query,
#     run_emb_preprocess,
#     run_emb_eval,
#     run_tx_infer,
#     run_tx_predict,
#     run_tx_preprocess_infer,
#     run_tx_preprocess_train,
#     run_tx_train,
# )

# # inspired by state https://github.com/ArcInstitute/state/tree/main/src/state

# def get_args() -> tuple[ap.Namespace, list[str]]:
#     """Parse known args and return remaining args for Hydra overrides"""
#     parser = ap.ArgumentParser()
#     subparsers = parser.add_subparsers(required=True, dest="command")
#     add_arguments_emb(subparsers.add_parser("emb"))
#     add_arguments_tx(subparsers.add_parser("tx"))

#     # Use parse_known_args to get both known args and remaining args
#     return parser.parse_args()


# def load_hydra_config(method: str, overrides: list[str] = None) -> DictConfig:
#     """Load Hydra config with optional overrides"""
#     if overrides is None:
#         overrides = []

#     # Initialize Hydra with the path to your configs directory
#     # Adjust the path based on where this file is relative to configs/
#     with initialize(version_base=None, config_path="configs"):
#         match method:
#             case "emb":
#                 cfg = compose(config_name="state-defaults", overrides=overrides)
#             case "tx":
#                 cfg = compose(config_name="config", overrides=overrides)
#             case _:
#                 raise ValueError(f"Unknown method: {method}")
#     return cfg


# def show_hydra_help(method: str):
#     """Show Hydra configuration help with all parameters"""
#     from omegaconf import OmegaConf

#     # Load the default config to show structure
#     cfg = load_hydra_config(method)

#     print("Hydra Configuration Help")
#     print("=" * 50)
#     print(f"Configuration for method: {method}")
#     print()
#     print("Full configuration structure:")
#     print(OmegaConf.to_yaml(cfg))
#     print()
#     print("Usage examples:")
#     print("  Override single parameter:")
#     print("    uv run state tx train data.batch_size=64")
#     print()
#     print("  Override nested parameter:")
#     print("    uv run state tx train model.kwargs.hidden_dim=512")
#     print()
#     print("  Override multiple parameters:")
#     print("    uv run state tx train data.batch_size=64 training.lr=0.001")
#     print()
#     print("  Change config group:")
#     print("    uv run state tx train data=custom_data model=custom_model")
#     print()
#     print("Available config groups:")

#     # Show available config groups
#     from pathlib import Path

#     config_dir = Path(__file__).parent / "configs"
#     if config_dir.exists():
#         for item in config_dir.iterdir():
#             if item.is_dir() and not item.name.startswith("."):
#                 configs = [f.stem for f in item.glob("*.yaml")]
#                 if configs:
#                     print(f"  {item.name}: {', '.join(configs)}")

#     exit(0)


# def main():
#     args = get_args()

#     match args.command: 
#         case "init":
#             pass
#         case "run":
#             pass
#         case "visualize":
#             pass

#     match args.command:
#         case "emb":
#             match args.subcommand:
#                 case "fit":
#                     cfg = load_hydra_config("emb", args.hydra_overrides)
#                     run_emb_fit(cfg, args)
#                 case "transform":
#                     run_emb_transform(args)
#                 case "query":
#                     run_emb_query(args)
#                 case "preprocess":
#                     run_emb_preprocess(args)
#                 case "eval":
#                     run_emb_eval(args)
#         case "tx":
#             match args.subcommand:
#                 case "train":
#                     if hasattr(args, "help") and args.help:
#                         # Show Hydra configuration help
#                         show_hydra_help("tx")
#                     else:
#                         # Load Hydra config with overrides for sets training
#                         cfg = load_hydra_config("tx", args.hydra_overrides)
#                         run_tx_train(cfg)
#                 case "predict":
#                     # For now, predict uses argparse and not hydra
#                     run_tx_predict(args)
#                 case "infer":
#                     # Run inference using argparse, similar to predict
#                     run_tx_infer(args)
#                 case "preprocess_train":
#                     # Run preprocessing using argparse
#                     run_tx_preprocess_train(args.adata, args.output, args.num_hvgs)
#                 case "preprocess_infer":
#                     # Run inference preprocessing using argparse
#                     run_tx_preprocess_infer(args.adata, args.output, args.control_condition, args.pert_col, args.seed)

from __future__ import annotations

import sys
from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

# Subcommands you want to support as: `storm run ...`
VALID_COMMANDS = {"run", "init", "visualize"}

def _rewrite_argv_for_hydra(argv: list[str]) -> list[str]:
    """
    Convert `storm <cmd> key=val ...` into Hydra-friendly overrides:
      `storm command=<cmd> key=val ...`

    If the user already provided `command=...`, we don't override it.
    """
    if len(argv) >= 2 and argv[1] in VALID_COMMANDS:
        cmd = argv[1]
        rest = [argv[0], *argv[2:]]
        if not any(a.startswith("command=") for a in rest[1:]):
            rest.insert(1, f"command={cmd}")
        return rest
    return argv

def _dispatch(cfg: DictConfig) -> None:
    """
    Route the composed Hydra config to the right implementation.
    Expect `cfg.command` to be set via config group or injected override.
    """
    cmd = str(cfg.get("command", "")).strip()
    if cmd not in VALID_COMMANDS:
        raise ValueError(
            f"Unknown command={cmd!r}. Valid: {sorted(VALID_COMMANDS)}. "
            f"Tip: use `storm run ...` or `storm command=run ...`."
        )

    # Import lazily so each subcommand can have its own heavier deps.
    if cmd == "run":
        from storm.cli.run import main as run_main
        run_main(cfg)
        return

    if cmd == "init":
        from storm.cli.init import main as init_main
        init_main(cfg)
        return

    if cmd == "visualize":
        from storm.cli.visualize import main as vis_main
        vis_main(cfg)
        return

    # Should be unreachable due to earlier validation.
    raise RuntimeError(f"Unhandled command: {cmd!r}")

@hydra.main(config_path="../configs", config_name="storm", version_base="1.3")
def _hydra_entry(cfg: DictConfig) -> None:
    # Optional: helpful debug
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))

    _dispatch(cfg)

def main() -> None:
    sys.argv = _rewrite_argv_for_hydra(sys.argv)
    _hydra_entry()

if __name__ == "__main__":
    main()