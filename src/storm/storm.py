import scanpy as sc
from perturbench.data.datasplitter import PerturbationDataSplitter
import numpy as np
import pandas as pd
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from save import save_datasets
from anndata import AnnData


def stratified_subsample_train(df, frac, group_keys, split_col="transfer_split_seed1", train_label="train", random_state=42):
    """
    Subsample only the training set rows, stratified by group_keys.
    Validation and test rows are kept unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with split assignments.
    frac : float
        Fraction of training rows to sample per group.
    group_keys : list[str]
        Column names to group by (e.g., condition, cell_class).
    split_col : str
        Column indicating train/val/test split.
    train_label : str
        Value in split_col corresponding to training rows.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataframe with downsampled training rows and unchanged val/test.
    """
    # Split by assignment
    train_df = df[df[split_col] == train_label]
    others_df = df[df[split_col] != train_label]

    # Stratified downsample of train
    train_down = (
        train_df
        .groupby(group_keys, group_keys=False, sort=False, observed=False)
        .sample(frac=frac, random_state=random_state)
    )

    # Combine train + others
    return pd.concat([train_down, others_df]).sort_index()

def manual_controls(df, condition_col='condition', control_value='ctrl'):
    """
    Manually assign train/val/test splits based on BioSamp values for control cells only.
    
    Parameters:
    - df: DataFrame with columns 'BioSamp', 'condition', and 'transfer_split_seed1'
    - condition_col: Name of the condition column (default: 'condition')
    - control_value: Value indicating control cells (default: 'ctrl')
    
    Returns:
    - df: DataFrame with updated 'transfer_split_seed1' column
    
    Split logic (applied only to control cells):
    - Batch 1 Sample 1 and 2 -> train
    - Batch 2 mouse 1 and 2 -> val
    - Batch 2 mouse 3 -> test
    """
    df = df.copy()
    
    # Create mask for control cells only
    control_mask = df[condition_col] == control_value
    
    # Train: Batch 1 Sample 1 and 2 (controls only)
    train_mask = control_mask & df['BioSamp'].str.contains('batch1_samp[12]', case=False, regex=True, na=False)
    df.loc[train_mask, 'transfer_split_seed1'] = 'train'
    
    # Val: Batch 2 mouse 1 and 2 (controls only)
    val_mask = control_mask & df['BioSamp'].str.contains('batch2_mouse[12]', case=False, regex=True, na=False)
    df.loc[val_mask, 'transfer_split_seed1'] = 'val'
    
    # Test: Batch 2 mouse 3 (controls only)
    test_mask = control_mask & df['BioSamp'].str.contains('batch2_mouse3', case=False, regex=True, na=False)
    df.loc[test_mask, 'transfer_split_seed1'] = 'test'
    
    return df

def check_coverage(
    df,
    condition_col: str = "condition",
    control_value: str = "ctrl",
    covariate_col: str = None,
    split_col: str = "transfer_split_seed1",
):
    """
    Manually assign train/val/test splits based on BioSamp values for control cells only,
    then check covariate coverage across splits and fix/drop problematic cell types.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'BioSamp', condition_col, covariate_col, and split_col.
    condition_col : str
        Name of the condition column (default: 'condition').
    control_value : str
        Value indicating control cells (default: 'ctrl').
    covariate_col : str
        Column name for cell types / covariates (e.g. covariate_keys[0]).
    split_col : str
        Column that stores split assignment (default: 'transfer_split_seed1').

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with split_col modified and some rows possibly removed.
    """
    df = df.copy()

    # ----------------- manual split for controls -----------------
    control_mask = df[condition_col] == control_value

    # Train: Batch 1 Sample 1 and 2 (controls only)
    train_mask = control_mask & df["BioSamp"].str.contains(
        "batch1_samp[12]", case=False, regex=True, na=False
    )
    df.loc[train_mask, split_col] = "train"

    # Val: Batch 2 mouse 1 and 2 (controls only)
    val_mask = control_mask & df["BioSamp"].str.contains(
        "batch2_mouse[12]", case=False, regex=True, na=False
    )
    df.loc[val_mask, split_col] = "val"

    # Test: Batch 2 mouse 3 (controls only)
    test_mask = control_mask & df["BioSamp"].str.contains(
        "batch2_mouse3", case=False, regex=True, na=False
    )
    df.loc[test_mask, split_col] = "test"

    # ----------------- covariate coverage checks -----------------
    if covariate_col is None:
        print("manual_controls: covariate_col is None, skipping covariate coverage checks.")
        return df

    if covariate_col not in df.columns:
        raise KeyError(f"manual_controls: covariate_col '{covariate_col}' not in DataFrame.")

    # Count examples per covariate per split
    split_counts = (
        df.groupby(covariate_col, observed=False)[split_col]
        .value_counts()
        .unstack(fill_value=0)
    )

    # Ensure train/val/test columns exist even if zero everywhere
    for split_name in ("train", "val", "test"):
        if split_name not in split_counts.columns:
            split_counts[split_name] = 0

    # Identify problematic covariates (before any fixes)
    zero_train_initial = split_counts[split_counts["train"] == 0].index.tolist()
    zero_val_initial = split_counts[split_counts["val"] == 0].index.tolist()
    zero_test_initial = split_counts[split_counts["test"] == 0].index.tolist()

    # Print summary before modifications
    if zero_train_initial:
        print("Cell types with 0 training examples (before removal):")
        for ct in zero_train_initial:
            print(f"  - {ct}")
    else:
        print("No cell types with 0 training examples.")

    print(f"Number of cell types with 0 training examples: {len(zero_train_initial)}")
    print(f"Number of cell types with 0 validation examples: {len(zero_val_initial)}")
    print(f"Number of cell types with 0 test examples: {len(zero_test_initial)}")

    # 1) Drop covariates with 0 training examples
    if zero_train_initial:
        print("Dropping cell types with 0 training examples from DataFrame:")
        for ct in zero_train_initial:
            print(f"  - dropping {ct}")
        df = df[~df[covariate_col].isin(zero_train_initial)].copy()

    df[covariate_col] = df[covariate_col].cat.remove_unused_categories()

    # 2) Recompute counts after dropping those, then fix val/test coverage
    if df.empty:
        print("DataFrame is empty after dropping cell types with 0 training examples.")
        return df

    control_df = df[df[condition_col] == control_value].copy()

    split_counts2 = (
        control_df.groupby(covariate_col, observed=False)[split_col]
        .value_counts()
        .unstack(fill_value=0)
    )
    for split_name in ("train", "val", "test"):
        if split_name not in split_counts2.columns:
            split_counts2[split_name] = 0

    zero_val_after_drop = split_counts2[split_counts2["val"] == 0].index.tolist()
    zero_test_after_drop = split_counts2[split_counts2["test"] == 0].index.tolist()

    # Union of covariates with no val or no test (but now all have some train)
    covariates_to_train_only = sorted(set(zero_val_after_drop) | set(zero_test_after_drop))

    if covariates_to_train_only:
        print(
            "Cell types with 0 validation or 0 test examples after dropping no-train types; "
            "setting all of their splits to 'train':"
        )
        for ct in covariates_to_train_only:
            print(f"  - {ct}")
        df.loc[df[covariate_col].isin(covariates_to_train_only), split_col] = "train"
    else:
        print("All remaining cell types have at least one example in val and test (or both).")

    return df

def keep_covariates_with_train_control(
    df: pd.DataFrame,
    adata=None,
    split_col: str = "transfer_split_seed1",
    perturbation_key: str = "Assign",
    control_value: str = "NT_0",
    covariate_key: str = "predicted.subclass",
):
    """
    Restrict to covariate groups (e.g., cell types) that contain at least one
    training-control example: (split == 'train') & (perturbation == control_value).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [split_col, perturbation_key, covariate_key].
    adata : anndata.AnnData or None
        If provided, prints before/after shapes using adata.obs[covariate_key].
        (No in-place changes to adata.)
    split_col : str
        Split column (default: 'transfer_split_seed1').
    perturbation_key : str
        Perturbation/condition column (e.g., 'Assign').
    control_value : str
        Control value (e.g., 'NT_0').
    covariate_key : str
        Covariate column (e.g., 'predicted.subclass').

    Returns
    -------
    df_filtered : pd.DataFrame
        Subset of df containing only rows whose covariate belongs to the valid set.
    valid_covariates : np.ndarray
        Array of covariate values that had at least one (train, control) example.
    """
    # Find covariates with at least one (train & control) example
    mask = (df[split_col] == "train") & (df[perturbation_key] == control_value)
    valid_covariates = df.loc[mask, covariate_key].dropna().unique()

    print(f"Cell types with train + {control_value}: {len(valid_covariates)}")
    print(f"Valid cell types: {valid_covariates}")

    # Filter df to keep only those covariates
    original_n = len(df)
    df_filtered = df[df[covariate_key].isin(valid_covariates)].copy()

    if adata is not None:
        orig_shape = adata.shape
        filt_shape_n = int((adata.obs[covariate_key].isin(valid_covariates)).sum())
        print(f"\nOriginal shape: {orig_shape}")
        print(f"Filtered shape: ({filt_shape_n}, {orig_shape[1]})")
        print(f"Removed {orig_shape[0] - filt_shape_n} cells")
    else:
        print(f"\nOriginal rows: {original_n}")
        print(f"Filtered rows: {len(df_filtered)}")
        print(f"Removed {original_n - len(df_filtered)} rows")

    return df_filtered, valid_covariates

def check_coverage_adata(
    adata: AnnData,
    condition_col: str = "condition",
    control_value: str = "ctrl",
    covariate_col: str = None,
    split_col: str = "transfer_split_seed1",
) -> AnnData:
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
        print("check_coverage_adata: covariate_col is None, skipping coverage checks.")
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
        print("Cell types with 0 training examples (before removal):")
        for ct in zero_train_initial:
            print(f"  - {ct}")
    else:
        print("No cell types with 0 training examples.")

    print(f"Number of cell types with 0 training examples: {len(zero_train_initial)}")
    print(f"Number of cell types with 0 validation examples: {len(zero_val_initial)}")
    print(f"Number of cell types with 0 test examples: {len(zero_test_initial)}")

    # 1) Drop covariates with 0 training examples (subset adata!)
    if zero_train_initial:
        print("Dropping cell types with 0 training examples from AnnData:")
        drop_mask = obs[covariate_col].isin(zero_train_initial)
        for ct in zero_train_initial:
            print(f"  - dropping {ct}")
        keep_mask = ~drop_mask.to_numpy()
        adata = adata[keep_mask].copy()
        obs = adata.obs  # refresh view after subsetting

    # Remove unused categories if categorical
    if pd.api.types.is_categorical_dtype(obs[covariate_col]):
        adata.obs[covariate_col] = obs[covariate_col].cat.remove_unused_categories()
        obs = adata.obs

    # 2) Recompute counts after dropping those, then fix val/test coverage
    if adata.n_obs == 0:
        print("AnnData has 0 cells after dropping no-train cell types.")
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

    if covariates_to_train_only:
        print(
            "Cell types with 0 validation or 0 test examples after dropping no-train types; "
            "setting all of their splits to 'train':"
        )
        mask_train_only = obs[covariate_col].isin(covariates_to_train_only)
        for ct in covariates_to_train_only:
            print(f"  - {ct}")
        adata.obs.loc[mask_train_only, split_col] = "train"
    else:
        print("All remaining cell types have at least one example in val and test (or both).")

    return adata

def create_dataset_variants(
        adata, 
        balanced_transfer_splitter, 
        perturbations, 
        perturbation_key, 
        covariate_key,
        control_value, 
        manual_control, 
        base_fractions=[1.0, 0.6, 0.2], 
    ):
    """
    Automates the creation of dataset variants with different perturbation exclusions/inclusions
    and different sampling fractions.
    
    Parameters:
    - adata: The original AnnData object
    - balanced_transfer_splitter: The splitter object with obs_dataframe
    - base_fractions: List of fractions to subsample (1.0 means full dataset)
    """
    
    # Define perturbation sets
    all_perturbations_to_remove = perturbations
    some_perturbations_to_remove = perturbations[:3]
    
    # Dataset configurations
    dataset_configs = {
        # Full dataset (no removals, but add back specific ones as train)
        '11': {
            'add_back_as_train': all_perturbations_to_remove
        },
        # Remove 3 perturbations, add back 3 as train
        '8': {
            'add_back_as_train': some_perturbations_to_remove
        },
        # Remove all 6 perturbations
        '5': {
            'add_back_as_train': []
        }
    }
    
    # Store all created datasets
    datasets = {}
    
    # Group columns for stratified sampling
    group_cols = [perturbation_key, covariate_key, "transfer_split_seed1"]
    
    # Create each dataset configuration
    for config_name, config in dataset_configs.items():
        print(f"Creating dataset configuration {config_name}...")
        
        df_config = balanced_transfer_splitter.obs_dataframe

        if manual_control:
            df_config = check_coverage(df_config, 
                                       condition_col=perturbation_key, 
                                       control_value=control_value,
                                       covariate_col=covariate_key)

        # Keep only covariate groups that have at least one (train & control) example
        # Assign == perturbation_key, control == "NT_0", covariate == "predicted.subclass"
        df_config, _valid = keep_covariates_with_train_control(
            df_config,
            adata=adata,
            split_col="transfer_split_seed1",
            perturbation_key=perturbation_key,
            control_value=control_value,
            covariate_key=covariate_key,
        )

        # Add back perturbations as training data
        if config['add_back_as_train']:
            # Get rows to add back 
            rows_to_add = adata.obs[adata.obs[perturbation_key].isin(config['add_back_as_train'])].copy()
            # Set them as training data
            rows_to_add["transfer_split_seed1"] = "train"
            # Combine with existing data
            df_config = pd.concat([df_config, rows_to_add], ignore_index=False)
        
        # Create subsampled versions for each fraction
        for frac in base_fractions:
            if frac == 1.0:
                frac_name = "100"
                df_sampled = df_config.copy()
            else:
                frac_name = str(int(frac * 100))
                # Use different random seeds for different fractions
                random_seed = 1 if frac == 0.6 else 2
                df_sampled = stratified_subsample_train(
                    df_config, 
                    frac=frac, 
                    group_keys=group_cols, 
                    random_state=random_seed
                )
            
            qual_name = "high" if frac_name == "100" else "medium" if frac_name == "60" else "low"
            amt_name = "high" if config_name == "11" else "medium" if config_name == "8" else "low"

            dataset_key = f"qual_{qual_name}_amt_{amt_name}"

            datasets[dataset_key] = df_sampled
            
            print(f"  Created {dataset_key} with {len(df_sampled)} cells")
    
    return datasets

@hydra.main(config_path="../configs", config_name="storm", version_base="1.3")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    
    covariate_keys = OmegaConf.to_container(cfg.covariates.names, resolve=True)

    main_dir = cfg.output.main_dir + '/'

    # Load data
    adata = sc.read_h5ad(cfg.adata.input_path)

    # filter all cell types that don't have training, val, test controls. 
    adata = check_coverage_adata(adata, 
                                 condition_col=cfg.perturbations.key, 
                                 control_value=cfg.perturbations.control_value, 
                                 covariate_col=covariate_keys[0])

    # Determine perturbations to remove
    if cfg.perturbations.get('randomize', False):
        # Get all unique perturbations
        all_perturbations = adata.obs[cfg.perturbations.key].unique().tolist()
        
        # Remove control from the list if it exists
        control_value = cfg.perturbations.get('control_value', 'ctrl')
        if control_value in all_perturbations:
            all_perturbations.remove(control_value)
        
        # Shuffle with seed
        seed = cfg.perturbations.get('random_seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        random.shuffle(all_perturbations)
        
        # Take top N perturbations to remove
        n_remove = cfg.perturbations.get('n_remove', 6)
        perturbations_to_remove = all_perturbations[:n_remove]
        
        print(f"Randomly selected perturbations to remove (seed={seed}): {perturbations_to_remove}")
    else:
        # Use manually specified list
        perturbations_to_remove = OmegaConf.to_container(cfg.perturbations.remove, resolve=True)
        print(f"Using specified perturbations to remove: {perturbations_to_remove}")

    df_initial = adata.obs[~adata.obs[cfg.perturbations.key].isin(perturbations_to_remove)]

    # Create and run splitter
    splitter = PerturbationDataSplitter(
        df_initial,
        perturbation_key=cfg.perturbations.key,
        covariate_keys=covariate_keys,
        perturbation_control_value=cfg.perturbations.control_value,
    )

    split = splitter.split_covariates(
        seed=cfg.splitter.seed,
        print_split=True,
        max_heldout_fraction_per_covariate=cfg.splitter.max_heldout_fraction_per_covariate,
        max_heldout_covariates=cfg.splitter.max_heldout_covariates,
    )

    # Create dataset variants
    datasets = create_dataset_variants(adata, splitter, perturbations_to_remove, cfg.perturbations.key, covariate_keys[0], cfg.perturbations.control_value, cfg.manual_control)

    # save all datasets
    save_datasets(
        datasets=datasets,
        adata=adata,
        dataset_name=cfg.dataset.name,
        perturbation_key=cfg.perturbations.key,
        covariate_key=covariate_keys[0],
        control_value=cfg.perturbations.control_value,
        main_dir=main_dir,
    )

    print("All datasets created and saved successfully!")

    # Save note as README.md
    if "note" in cfg and cfg.note:
        readme_path = os.path.join(main_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(cfg.note.strip() + "\n")
        print(f"Note saved to {readme_path}")

# Main execution
if __name__ == "__main__":
    main()