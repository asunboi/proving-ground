import scanpy as sc
import numpy as np
import pandas as pd
import os
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import random
from anndata import AnnData
import logging
from storm.save import save_datasets
from storm.scale import create_scaled_datasets
from storm.splitter import PerturbationDataSplitter, apply_toml_manual_split, apply_csv_manual_split
import scipy.sparse as sp

# module-level logger
log = logging.getLogger(__name__)

def choose_perturbations_to_remove(adata, perturbation_key, perturb_cfg) -> list[str]:
    if perturb_cfg.get("randomize", False):
        all_perturbations = adata.obs[perturbation_key].unique().tolist()
        control_value = perturb_cfg.get("control_value", "ctrl")
        if control_value in all_perturbations:
            all_perturbations.remove(control_value)
        seed = perturb_cfg.get("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        random.shuffle(all_perturbations)
        n_remove = perturb_cfg.get("n_remove", 6)
        perturbations_to_remove = all_perturbations[:n_remove]
    else:
        perturbations_to_remove = OmegaConf.to_container(perturb_cfg.remove, resolve=True)
    return perturbations_to_remove

def normalize_seeds(seed_cfg):
    """
    Turn cfg.splitter.seed into an iterable of ints.

    Handles:
      - int
      - list / tuple / range
      - OmegaConf ListConfig
      - optional string like 'range(1,11)'
    """
    # Optional: parse 'range(a,b,step)' style strings
    if isinstance(seed_cfg, str) and seed_cfg.startswith("range(") and seed_cfg.endswith(")"):
        inner = seed_cfg[len("range("):-1]
        parts = [int(x.strip()) for x in inner.split(",") if x.strip()]

        if len(parts) == 1:
            return range(parts[0])
        elif len(parts) == 2:
            return range(parts[0], parts[1])
        elif len(parts) == 3:
            return range(parts[0], parts[1], parts[2])
        else:
            raise ValueError(f"Unsupported range spec: {seed_cfg}")

    # Already some kind of iterable container
    if isinstance(seed_cfg, (ListConfig, list, tuple, range)):
        return [int(s) for s in seed_cfg]

    # Scalar → wrap into list
    return [int(seed_cfg)]

def main(cfg: DictConfig):

    log.info(OmegaConf.to_yaml(cfg))

    # Load data
    adata = sc.read_h5ad(cfg.data.adata_path)

    # convert to sparse csr once to fit format to perturbench batch
    adata.X = adata.X.tocsr()

    # HACK: DROP ALL CT WITH LESS THAN X CELLS
    # NOTE: why am I doing this? was it because of the splitter misbehaving?
    counts = adata.obs.groupby(cfg.data.covariate_key, dropna=False)[cfg.data.covariate_key].transform("size")
    mask = counts.ge(50)
    adata = adata[mask].copy()

    # # REFACTOR: put this into scale.py
    # # Determine perturbations to remove
    if cfg.scale.enabled:
        perturbations_to_remove = choose_perturbations_to_remove(
            adata=adata,
            perturbation_key=cfg.perturbations.key,
            perturb_cfg=cfg.perturbations,
        )
        df_initial = adata.obs[~adata.obs[cfg.perturbations.key].isin(perturbations_to_remove)]
    else:
        perturbations_to_remove = []
        df_initial = adata.obs

    # Create and run splitter
    splitter = PerturbationDataSplitter(
        df_initial,
        perturbation_key=cfg.data.perturbation_key,
        covariate_keys=cfg.data.covariate_key,
        batch_key=cfg.data.batch_key,
        perturbation_control_value=cfg.data.control_value,
    )

    seeds = normalize_seeds(cfg.splitter.seed)

    ### checks if manual split, and where the manual split source is from to set the function
    MANUAL_SPLIT_DISPATCH = {
        "toml_path": apply_toml_manual_split,
        "csv_path": apply_csv_manual_split,
        # future: "json_path": apply_json_manual_split,
    }
    if cfg.splitter.manual:
        provided = [
            (field, getattr(cfg.splitter, field))
            for field in MANUAL_SPLIT_DISPATCH
            if getattr(cfg.splitter, field)
        ]
        if len(provided) != 1:
            raise ValueError(
                f"Exactly one manual split source must be set, got: {[k for k, _ in provided]}"
            )
        field, path = provided[0]
        split_fn = MANUAL_SPLIT_DISPATCH[field]
        df_initial = split_fn(
            df_initial,
            cfg,
            perturbation_suffix="_0",
        )
    else:
        for seed in seeds:
            splitter.split_covariates(
                seed=seed,
                print_split=True,
                max_heldout_fraction_per_covariate=cfg.splitter.max_heldout_fraction_per_covariate,
                max_heldout_covariates=cfg.splitter.max_heldout_covariates,
            )

    # if cfg.scale.enabled = true, returns scaled dict. if not, returns {full: adata.obs}
    datasets = create_scaled_datasets(
        adata=adata,
        splitter=splitter,
        perturbations_to_remove=perturbations_to_remove,
        perturbation_key=cfg.data.perturbation_key,
        covariate_key=cfg.data.covariate_key,
        control_value=cfg.data.control_value,
        manual_control=cfg.splitter.manual_control,
        base_fractions=cfg.scale.base_fractions,
        enable=cfg.scale.enabled,
    )

    save_datasets(
        datasets=datasets,
        adata=adata,
        cfg=cfg,
    )

    log.info("All datasets created and saved successfully!")

    # Save note as README.md
    if "note" in cfg and cfg.note:
        readme_path = os.path.join(cfg.output.main_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(cfg.note.strip() + "\n")
        log.info(f"Note saved to {readme_path}")

# Main execution
if __name__ == "__main__":
    main()