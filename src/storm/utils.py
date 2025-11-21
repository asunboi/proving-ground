import logging

log = logging.getLogger(__name__)

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
        log.info("manual_controls: covariate_col is None, skipping covariate coverage checks.")
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
        log.info("Cell types with 0 training examples (before removal):")
        for ct in zero_train_initial:
            log.info(f"  - {ct}")
    else:
        log.info("No cell types with 0 training examples.")

    log.info(f"Number of cell types with 0 training examples: {len(zero_train_initial)}")
    log.info(f"Number of cell types with 0 validation examples: {len(zero_val_initial)}")
    log.info(f"Number of cell types with 0 test examples: {len(zero_test_initial)}")

    # 1) Drop covariates with 0 training examples
    if zero_train_initial:
        log.info("Dropping cell types with 0 training examples from DataFrame:")
        for ct in zero_train_initial:
            log.info(f"  - dropping {ct}")
        df = df[~df[covariate_col].isin(zero_train_initial)].copy()

    df[covariate_col] = df[covariate_col].cat.remove_unused_categories()

    # 2) Recompute counts after dropping those, then fix val/test coverage
    if df.empty:
        log.info("DataFrame is empty after dropping cell types with 0 training examples.")
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
        log.info(
            "Cell types with 0 validation or 0 test examples after dropping no-train types; "
            "setting all of their splits to 'train':"
        )
        for ct in covariates_to_train_only:
            log.info(f"  - {ct}")
        df.loc[df[covariate_col].isin(covariates_to_train_only), split_col] = "train"
    else:
        log.info("All remaining cell types have at least one example in val and test (or both).")

    return df