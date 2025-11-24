import scanpy as sc
import numpy as np
import pandas as pd
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from save import save_datasets
from anndata import AnnData
from scale import create_scaled_datasets
import logging
from splitter import PerturbationDataSplitter, apply_toml_manual_split

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

def check_coverage_adata(
    adata: AnnData,
    condition_col: str = "condition",
    control_value: str = "ctrl",
    covariate_col: str = None,
    split_col: str = "transfer_split_seed1",
) -> tuple[AnnData, list[str]]:
    """
    Apply manual control splitting and coverage checks directly on an AnnData object.

    - Manually assign train/val/test for control cells based on `BioSamp`.
    - Drop cell types (covariate levels) that have 0 training examples.
    - For remaining cell types that have 0 val or 0 test, set all their splits to 'train'.
    - Subset `adata` so X/obs/var all stay consistent.

    Parameters
    ----------
    adata : AnnData
        AnnData with .obs columns: 'BioSamp', condition_col, covariate_col, split_col.
    condition_col : str
        Name of the condition column (default: 'condition').
    control_value : str
        Value indicating control cells (default: 'ctrl').
    covariate_col : str
        Column name for cell types / covariates (e.g. 'predicted.subclass').
    split_col : str
        Column that stores split assignment (default: 'transfer_split_seed1').

    Returns
    -------
    AnnData
        New AnnData with updated obs and (possibly) fewer cells.
    """
    # Work on a copy so we don't surprise-callers by mutating in place
    adata = adata.copy()
    obs = adata.obs

    if covariate_col is None:
        log.info("check_coverage_adata: covariate_col is None, skipping coverage checks.")
        return adata
    if covariate_col not in obs.columns:
        raise KeyError(f"check_coverage_adata: covariate_col '{covariate_col}' not in adata.obs")

    # ----------------- manual split for controls -----------------
    control_mask = obs[condition_col] == control_value

    # Train: Batch 1 Sample 1 and 2 (controls only)
    train_mask = control_mask & obs["BioSamp"].str.contains(
        "batch1_samp[12]", case=False, regex=True, na=False
    )
    obs.loc[train_mask, split_col] = "train"

    # Val: Batch 2 mouse 1 and 2 (controls only)
    val_mask = control_mask & obs["BioSamp"].str.contains(
        "batch2_mouse[12]", case=False, regex=True, na=False
    )
    obs.loc[val_mask, split_col] = "val"

    # Test: Batch 2 mouse 3 (controls only)
    test_mask = control_mask & obs["BioSamp"].str.contains(
        "batch2_mouse3", case=False, regex=True, na=False
    )
    obs.loc[test_mask, split_col] = "test"

    # ----------------- covariate coverage checks -----------------
    split_counts = (
        obs.groupby(covariate_col, observed=False)[split_col]
        .value_counts()
        .unstack(fill_value=0)
    )

    # Ensure train/val/test columns exist even if zero everywhere
    for split_name in ("train", "val", "test"):
        if split_name not in split_counts.columns:
            split_counts[split_name] = 0

    zero_train_initial = split_counts[split_counts["train"] == 0].index.tolist()
    zero_val_initial = split_counts[split_counts["val"] == 0].index.tolist()
    zero_test_initial = split_counts[split_counts["test"] == 0].index.tolist()

    # Print summary before modifications
    if zero_train_initial:
        log.info("Cell types with 0 training examples (before removal):")
        for ct in zero_train_initial:
            log.info(f"  - {ct}")
    else:
        log.info("No cell types with 0 training examples.")

    log.info(f"Number of cell types with 0 training examples: {len(zero_train_initial)}")
    log.info(f"Number of cell types with 0 validation examples: {len(zero_val_initial)}")
    log.info(f"Number of cell types with 0 test examples: {len(zero_test_initial)}")

    # 1) Drop covariates with 0 training examples (subset adata!)
    if zero_train_initial:
        log.info("Dropping cell types with 0 training examples from AnnData:")
        drop_mask = obs[covariate_col].isin(zero_train_initial)
        for ct in zero_train_initial:
            log.info(f"  - dropping {ct}")
        keep_mask = ~drop_mask.to_numpy()
        adata = adata[keep_mask].copy()
        obs = adata.obs  # refresh view after subsetting

    # Remove unused categories if categorical
    if pd.api.types.is_categorical_dtype(obs[covariate_col]):
        adata.obs[covariate_col] = obs[covariate_col].cat.remove_unused_categories()
        obs = adata.obs

    # 2) Recompute counts after dropping those, then fix val/test coverage
    if adata.n_obs == 0:
        log.info("AnnData has 0 cells after dropping no-train cell types.")
        return adata

    split_counts2 = (
        obs.groupby(covariate_col, observed=False)[split_col]
        .value_counts()
        .unstack(fill_value=0)
    )
    for split_name in ("train", "val", "test"):
        if split_name not in split_counts2.columns:
            split_counts2[split_name] = 0

    zero_val_after_drop = split_counts2[split_counts2["val"] == 0].index.tolist()
    zero_test_after_drop = split_counts2[split_counts2["test"] == 0].index.tolist()

    covariates_to_train_only = sorted(set(zero_val_after_drop) | set(zero_test_after_drop))

    # REFACTOR: put this into the splitter
    # All covariates still present after previous filtering
    remaining_covariates = sorted(obs[covariate_col].unique())

    # Those that are NOT in covariates_to_train_only
    covariates_holdout = sorted(
        set(remaining_covariates) - set(covariates_to_train_only)
    )

    if covariates_to_train_only:
        log.info(
            "Cell types with 0 validation or 0 test examples after dropping no-train types; "
            "setting all of their splits to 'train':"
        )
        mask_train_only = obs[covariate_col].isin(covariates_to_train_only)
        for ct in covariates_to_train_only:
            log.info(f"  - {ct}")
        adata.obs.loc[mask_train_only, split_col] = "train"
    else:
        log.info("All remaining cell types have at least one example in val and test (or both).")

    return adata, covariates_holdout

@hydra.main(config_path="../configs", config_name="storm", version_base="1.3")
def main(cfg: DictConfig):

    log.info(OmegaConf.to_yaml(cfg))
    
    covariate_keys = OmegaConf.to_container(cfg.covariates.names, resolve=True)

    # REFACTOR: don't declare a new variable / bloat just to add '/'
    main_dir = cfg.output.main_dir + '/'

    # Load data
    adata = sc.read_h5ad(cfg.adata.input_path)

    # REFACTOR: shouldn't have to get holdout covariates like this, integrate into splitter.
    # BUG: this probably still wouldn't work because the below function that uses covariates_holdout doesn't fix the control issue.
    # filter all cell types that don't have training, val, test controls. 
    adata, covariates_holdout = check_coverage_adata(adata, 
                                 condition_col=cfg.perturbations.key, 
                                 control_value=cfg.perturbations.control_value, 
                                 covariate_col=covariate_keys[0])

    # # REFACTOR: put this into scale.py
    # # Determine perturbations to remove
    # if cfg.perturbations.get('randomize', False):
    #     # Get all unique perturbations
    #     all_perturbations = adata.obs[cfg.perturbations.key].unique().tolist()
        
    #     # Remove control from the list if it exists
    #     control_value = cfg.perturbations.get('control_value', 'ctrl')
    #     if control_value in all_perturbations:
    #         all_perturbations.remove(control_value)
        
    #     # Shuffle with seed
    #     seed = cfg.perturbations.get('random_seed', 42)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     random.shuffle(all_perturbations)
        
    #     # Take top N perturbations to remove
    #     n_remove = cfg.perturbations.get('n_remove', 6)
    #     perturbations_to_remove = all_perturbations[:n_remove]
        
    #     print(f"Randomly selected perturbations to remove (seed={seed}): {perturbations_to_remove}")
    # else:
    #     # Use manually specified list
    #     perturbations_to_remove = OmegaConf.to_container(cfg.perturbations.remove, resolve=True)
    #     print(f"Using specified perturbations to remove: {perturbations_to_remove}")

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
        perturbation_key=cfg.perturbations.key,
        covariate_keys=covariate_keys,
        perturbation_control_value=cfg.perturbations.control_value,
    )

    if cfg.splitter.manual:
        apply_toml_manual_split(
            df_initial,
            cfg.splitter.toml_path,
            perturbation_suffix="_0",
        ) 
    else:
        splitter.split_covariates(
            seed=cfg.splitter.seed,
            print_split=True,
            max_heldout_fraction_per_covariate=cfg.splitter.max_heldout_fraction_per_covariate,
            max_heldout_covariates=cfg.splitter.max_heldout_covariates,
        )

    # # FIX: using perturbench's manual splitter with specified set of holdout covariates, testing to see if this works. 
    # # BUG: different behavior than split covariates, currently doesn't output any splits and causes error due to empty holdout
    # print(covariates_holdout)
    # splitter.split_covariates_manual(
    #     seed=cfg.splitter.seed,
    #     covariates_holdout=covariates_holdout,
    #     print_split=True,
    #     max_heldout_fraction_per_covariate=cfg.splitter.max_heldout_fraction_per_covariate,
    # )

    # Create dataset variants
    # datasets = create_dataset_variants(adata, splitter, perturbations_to_remove, cfg.perturbations.key, covariate_keys[0], cfg.perturbations.control_value, cfg.manual_control)

    # if cfg.scale.enabled = true, returns scaled dict. if not, returns {full: adata.obs}
    datasets = create_scaled_datasets(
        adata=adata,
        splitter=splitter,
        perturbations_to_remove=perturbations_to_remove,
        perturbation_key=cfg.perturbations.key,
        covariate_key=covariate_keys[0],
        control_value=cfg.perturbations.control_value,
        manual_control=cfg.splitter.manual_control,
        base_fractions=cfg.scale.base_fractions,
        enable=cfg.scale.enabled,
    )

    # # save all datasets
    # save_datasets(
    #     datasets=datasets,
    #     adata=adata,
    #     dataset_name=cfg.dataset.name,
    #     perturbation_key=cfg.perturbations.key,
    #     covariate_key=covariate_keys[0],
    #     control_value=cfg.perturbations.control_value,
    #     main_dir=main_dir,
    # )

    save_datasets(
        datasets=datasets,
        adata=adata,
        dataset_name=cfg.dataset.name,
        perturbation_key=cfg.perturbations.key,
        covariate_key=covariate_keys[0],
        control_value=cfg.perturbations.control_value,
        main_dir=main_dir,
        models=cfg.models
    )

    log.info("All datasets created and saved successfully!")

    # Save note as README.md
    if "note" in cfg and cfg.note:
        readme_path = os.path.join(main_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(cfg.note.strip() + "\n")
        log.info(f"Note saved to {readme_path}")

# Main execution
if __name__ == "__main__":
    main()