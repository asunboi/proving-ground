# src/perturbench/scale.py (path/name as appropriate)
from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Sequence
import numpy as np
import pandas as pd
from anndata import AnnData
import logging
from utils import check_coverage

# module-level logger
log = logging.getLogger(__name__)

def stratified_subsample_train(
    df: pd.DataFrame,
    frac: float,
    group_keys: Sequence[str],
    split_col: str = "transfer_split_seed1",
    train_label: str = "train",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Subsample only the training set rows, stratified by group_keys.
    Validation and test rows are kept unchanged.

    Parameters
    ----------
    df
        Input dataframe with split assignments.
    frac
        Fraction of training rows to sample per group.
    group_keys
        Column names to group by (e.g., condition, cell_class).
    split_col
        Column indicating train/val/test split.
    train_label
        Value in split_col corresponding to training rows.
    random_state
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataframe with downsampled training rows and unchanged val/test.
    """
    # Split by assignment
    train_df = df[df[split_col] == train_label]
    others_df = df[df[split_col] != train_label]

    if frac >= 1.0:
        # No downsampling requested; return original
        train_down = train_df
    else:
        # Stratified downsample of train
        train_down = (
            train_df
            .groupby(group_keys, group_keys=False, sort=False, observed=False)
            .sample(frac=frac, random_state=random_state)
        )

    # Combine train + others
    return pd.concat([train_down, others_df]).sort_index()

def _make_dataset_configs(perturbations: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Internal helper to define the '11', '8', '5' dataset configs.
    """
    all_perturbations_to_remove = list(perturbations)
    some_perturbations_to_remove = all_perturbations_to_remove[:3]

    return {
        # Full dataset (no removals, but add back specific ones as train)
        "11": {"add_back_as_train": all_perturbations_to_remove},
        # Remove 3 perturbations, add back 3 as train
        "8": {"add_back_as_train": some_perturbations_to_remove},
        # Remove all perturbations, no add-back
        "5": {"add_back_as_train": []},
    }

def _fraction_to_names(config_name: str, frac: float) -> tuple[str, str]:
    """
    Map (config, frac) to your qual/amt labels.
    """
    if frac == 1.0:
        qual_name = "high"
    elif np.isclose(frac, 0.6):
        qual_name = "medium"
    else:
        qual_name = "low"

    if config_name == "11":
        amt_name = "high"
    elif config_name == "8":
        amt_name = "medium"
    else:
        amt_name = "low"

    return qual_name, amt_name

def create_scaled_datasets(
    adata: AnnData,
    splitter,
    *,
    perturbations_to_remove: Sequence[str],
    perturbation_key: str,
    covariate_key: str,
    control_value: str,
    manual_control: bool,
    base_fractions: Sequence[float] = (1.0, 0.6, 0.2),
    enable: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Create a family of scaled datasets (as obs DataFrames) based on an existing splitter
    and a set of perturbations that were removed from the initial split.

    If `enable` is False, this becomes a no-op and just returns:
        {"full": adata.obs}

    Parameters
    ----------
    adata
        Original AnnData object (full data before scaling).
    splitter
        PerturbationDataSplitter instance that has already run `split_covariates`.
        Must expose `.obs_dataframe` with 'transfer_split_seed1' column.
    perturbations_to_remove
        Sequence of perturbation ids that were excluded from the initial split.
    perturbation_key
        Column in `adata.obs` giving perturbation labels.
    covariate_key
        Column in `adata.obs` giving the covariate (e.g. cell type).
    control_value
        Control condition value in `perturbation_key`.
    manual_control
        If True, re-check coverage via `check_coverage` before scaling.
    base_fractions
        Fractions used for downsampling within each dataset config.
        (1.0 means full dataset, no downsampling.)
    enable
        If False, disable all scaling and return a single full dataset.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from dataset key (e.g. 'qual_high_amt_high') to obs DataFrames.
    """
    if not enable:
        # No scaling; just return the full obs as a single entry.
        # copy() is optional depending on whether you want isolation.
        df_config = adata.obs.copy()
        if manual_control:
            df_config = check_coverage(
                df_config,
                condition_col=perturbation_key,
                control_value=control_value,
                covariate_col=covariate_key,
            )
        return {"full": df_config}

    dataset_configs = _make_dataset_configs(perturbations_to_remove)

    # Group columns for stratified sampling
    # Note: split_col is handled separately by `stratified_subsample_train`.
    group_cols = [perturbation_key, covariate_key, "transfer_split_seed1"]

    datasets: Dict[str, pd.DataFrame] = {}

    for config_name, config in dataset_configs.items():
        log.info(f"Creating dataset configuration {config_name}...")
        # Start from the post-split obs dataframe
        df_config = splitter.obs_dataframe.copy()

        # REFACTOR: this should eventually be something that the splitter takes into account / solves
        if manual_control:
            df_config = check_coverage(
                df_config,
                condition_col=perturbation_key,
                control_value=control_value,
                covariate_col=covariate_key,
            )

        # Add back perturbations as training data (if requested)
        add_back = config.get("add_back_as_train", [])
        if add_back:
            rows_to_add = adata.obs[adata.obs[perturbation_key].isin(add_back)].copy()
            rows_to_add["transfer_split_seed1"] = "train"
            df_config = pd.concat([df_config, rows_to_add], ignore_index=False)

        # Create subsampled versions for each fraction
        for frac in base_fractions:
            if frac == 1.0:
                df_sampled = df_config.copy()
            else:
                # Use different random seeds for different fractions (keeps behavior stable)
                if np.isclose(frac, 0.6):
                    random_seed = 1
                else:
                    random_seed = 2

                df_sampled = stratified_subsample_train(
                    df_config,
                    frac=frac,
                    group_keys=group_cols,
                    random_state=random_seed,
                )

            qual_name, amt_name = _fraction_to_names(config_name, frac)
            dataset_key = f"qual_{qual_name}_amt_{amt_name}"

            # Store the sampled obs dataframe directly
            datasets[dataset_key] = df_sampled

            log.info(f"  Created {dataset_key} with {len(df_sampled)} cells")

    return datasets