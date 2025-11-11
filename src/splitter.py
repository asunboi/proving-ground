import scanpy as sc
from perturbench.data.datasplitter import PerturbationDataSplitter
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import random


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
        train_df.groupby(group_keys, group_keys=False, sort=False)
        .apply(lambda g: g.sample(frac=frac, random_state=random_state))
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

def plot_counts(df, outfile, perturbation_key, covariate_key, seed_col="transfer_split_seed1"):
    """
    Make heatmaps of counts per (cell_line, gene) × seed from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least [cell_line_col, gene_col, seed_col].
    covariate_key : str, default "cell_line"
        Column containing cell line identifiers.
    perturbation_key : str, default "gene"
        Column containing gene identifiers.
    seed_col : str, default "transfer_split_seed1"
        Column containing split/seed values.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes
    """
    # counts
    count_long = (
        df.groupby([perturbation_key, seed_col, covariate_key])
        .size()
        .reset_index(name="count")
    )

    # pivot to wide format
    cell_counts_all = count_long.pivot(
        index=[covariate_key, perturbation_key],
        columns=seed_col,
        values="count"
    ).reset_index()

    cell_counts_all.columns.name = ""

    # drop rows with all NaN counts
    filtered = cell_counts_all.dropna(
        axis=0,
        how="all",
        subset=cell_counts_all.columns[2:]
    )

    cell_classes = filtered[covariate_key].unique()
    n_cols = 2
    n_rows = (len(cell_classes) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, cell_class in enumerate(cell_classes):
        ax = axes[i]
        subset = (
            filtered[filtered[covariate_key] == cell_class]
            .set_index(perturbation_key)
            .drop(columns=covariate_key)
            .fillna(0)
        )
        data = subset.values

        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_title(str(cell_class))
        ax.set_xticks(np.arange(subset.shape[1]))
        ax.set_xticklabels(subset.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(subset.shape[0]))
        ax.set_yticklabels(subset.index)

        # annotate each cell
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                value = data[y, x]
                ax.text(
                    x, y, int(value),
                    ha="center", va="center",
                    color="white" if value < data.max() / 2 else "black",
                    fontsize=8
                )

        fig.colorbar(im, ax=ax, shrink=0.6)

    # delete unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

    return fig, axes


def create_dataset_variants(adata, balanced_transfer_splitter, perturbations, perturbation_key, covariate_key, manual_control, base_fractions=[1.0, 0.6, 0.2]):
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
            df_config = manual_controls(df_config)

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

def generate_toml_config(
    dataset_name: str,
    h5ad_path: str,
    df: pd.DataFrame,
    perturbation_key: str = "condition",
    covariate_key: str = "cell_class",
    split_col: str = "transfer_split_seed1",
    train_label: str = "train",
    control_value: str = "ctrl",
) -> str:
    """
    Build a TOML config like:

    [datasets]
    boli_ctx = "/path/to/file.h5ad"

    [training]
    boli_ctx = "train"

    [zeroshot]

    [fewshot."boli_ctx.L6 CT CTX"]
    val = ["Xpo7"]
    test = ["Tbr1", "Satb2"]

    - One [fewshot."<dataset>.<celltype>"] table per unique covariate value.
    - 'val'/'test' are unique perturbations in those splits for that cell type.
    """

    def _escape_key(s: str) -> str:
        # Escape quotes/backslashes for a TOML quoted table key
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _fmt_array(items: "list[str]") -> "str":
        items = sorted({str(x) for x in items if pd.notna(x) and str(x).strip() != ""})
        return "[ " + ", ".join(f'"{x}"' for x in items) + " ]"

    lines = []

    # [datasets]
    lines.append("[datasets]")
    lines.append(f'{dataset_name} = "{h5ad_path}"')
    lines.append("")

    # [training]
    lines.append("[training]")
    lines.append(f'{dataset_name} = "{train_label}"')
    lines.append("")

    # [zeroshot] (left empty unless you later decide to populate)
    lines.append("[zeroshot]")
    lines.append("")

    # fewshot blocks per cell type
    if covariate_key in df.columns and perturbation_key in df.columns and split_col in df.columns:
        control_values = {control_value} if isinstance(control_value, str) else set(control_value)

        for cov_value, sub in df.groupby(covariate_key, sort=False):
            # filter out control entries
            val_mask  = (sub[split_col] == "val")  & (~sub[perturbation_key].isin(control_values))
            test_mask = (sub[split_col] == "test") & (~sub[perturbation_key].isin(control_values))

            val_list  = sub.loc[val_mask,  perturbation_key].tolist()
            test_list = sub.loc[test_mask, perturbation_key].tolist()
            # Skip if neither val nor test have items
            if not val_list and not test_list:
                continue

            table_key = f'{dataset_name}.{cov_value}'
            lines.append(f'[fewshot."{_escape_key(table_key)}"]')
            lines.append(f"val = {_fmt_array(val_list)}")
            lines.append(f"test = {_fmt_array(test_list)}")
            lines.append("")

    return "\n".join(lines).rstrip()  # clean trailing newline

def generate_yaml_config(dataset_name, split_name, h5ad_path, perturbation_key, covariate_key, control_value, csv_path):
    """
    Generate YAML configuration content for a dataset
    
    Parameters:
    - split_name: Name of the dataset (e.g., "100_8")
    - h5ad_path: Full path to the H5AD file
    - csv_path: Full path to the CSV split file
    
    Returns:
    - yaml_content: String containing the YAML configuration
    """
    
    yaml_template = f"""# @package _global_

defaults:
- override /model: latent_additive
- override /callbacks: default
- override /data: boli_ctx

data:
    datapath: {h5ad_path}
    covariate_keys: ["{covariate_key}"]
    perturbation_key: {perturbation_key}
    perturbation_combination_delimiter: +
    perturbation_control_value: {control_value}
    evaluation:
        split_value_to_evaluate: test
    splitter: 
        split_path: {csv_path}

# output directory, generated dynamically on each run
hydra:
    run:
        dir: ${{paths.log_dir}}/${{task_name}}/runs/${{now:%Y-%m-%d}}_${{now:%H-%M-%S}}_{dataset_name}_{split_name}
    sweep:
            dir: ${{paths.log_dir}}/${{task_name}}/multiruns/${{now:%Y-%m-%d}}_${{now:%H-%M-%S}}
            subdir: ${{hydra.job.num}}_{dataset_name}_{split_name}

    """
    
    return yaml_template

def generate_prediction_dataframe(df, split_col='transfer_split_seed1', 
                              condition_col='condition', 
                              cell_type_col='cell_class',
                              output_csv='test_combinations.csv'):
    """
    Extract unique condition and cell_type combinations from test split.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with split assignments
    split_col : str
        Column name for train/val/test split (default: 'transfer_split_seed1')
    condition_col : str
        Column name for perturbation condition (default: 'condition')
    cell_type_col : str
        Column name for cell type (default: 'cell_class')
    output_csv : str
        Output CSV filename (default: 'test_combinations.csv')
    
    Returns:
    --------
    test_combos : pd.DataFrame
        DataFrame with unique condition-cell_type combinations from test set
    """
    # Filter to test set only
    test_df = df[df[split_col] == 'test']
    
    # Get unique combinations of condition and cell_type
    test_combos = test_df[[condition_col, cell_type_col]].drop_duplicates()
    
    # Save to CSV
    test_combos.to_csv(output_csv, index=False)
    
    print(f"Extracted {len(test_combos)} unique test combinations")
    print(f"Saved to: {output_csv}")
    
    return test_combos

def save_datasets(datasets, adata, dataset_name, perturbation_key, covariate_key, control_value, main_dir):
    """
    Saves all datasets as CSV files and H5AD files, and generates corresponding YAML config files
    
    Parameters:
    - datasets: Dictionary of dataset name to dataframe
    - adata: The original AnnData object 
    - output_dir: Directory to save H5AD files
    - csv_dir: Directory to save CSV files
    - yaml_dir: Directory to save YAML files (defaults to csv_dir if None)
    """
    
    output_dir = main_dir + "data/"
    csv_dir = main_dir + "splits/"
    fig_dir = main_dir + "figures/"
    yaml_dir = main_dir + "cfg/"
    toml_dir = main_dir + "toml/"
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(yaml_dir, exist_ok=True)
    os.makedirs(toml_dir, exist_ok=True)

    for split_name, df in datasets.items():
        print(f"Saving {split_name}...")
        
        # Save CSV with transfer split information
        out = df[["transfer_split_seed1"]].copy()
        csv_filename = f"{dataset_name}_{split_name}_split.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        out.to_csv(csv_path, index=True, index_label="cell_barcode", header=False)

        # Generate the prediction dataframe for testing set
        prediction_filename = "prediction_dataframe.csv"
        prediction_path = os.path.join(main_dir, prediction_filename)
        generate_prediction_dataframe(df, condition_col=perturbation_key, cell_type_col=covariate_key, output_csv=prediction_path)
        
        # Plot split
        fig_filename = f"{dataset_name}_{split_name}_split.png"
        fig_path = os.path.join(fig_dir, fig_filename)
        plot_counts(df, fig_path, perturbation_key, covariate_key)

        # Save H5AD file
        adata_subset = adata[df.index].copy()
        h5ad_filename = f"{dataset_name}_{split_name}.h5ad"
        h5ad_path = os.path.join(output_dir, h5ad_filename)
        adata_subset.write_h5ad(h5ad_path)
        
        # Generate and save YAML config file
        yaml_content = generate_yaml_config(dataset_name, split_name, h5ad_path, perturbation_key, covariate_key, control_value, csv_path)
        yaml_filename = f"{dataset_name}_{split_name}.yaml"
        yaml_path = os.path.join(yaml_dir, yaml_filename)
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        # Generate and save TOML config file
        toml_content = generate_toml_config(
            dataset_name=dataset_name,
            h5ad_path=h5ad_path,
            df=df,
            perturbation_key=perturbation_key,
            covariate_key=covariate_key,
            split_col="transfer_split_seed1",
            train_label="train",
            control_value=control_value,
        )
        toml_filename = f"{dataset_name}_{split_name}.toml"
        toml_path = os.path.join(toml_dir, toml_filename)
        with open(toml_path, "w") as f:
            f.write(toml_content)

        # Create symlink in another directory
        link_dir = "/gpfs/home/asun/jin_lab/perturbench/src/perturbench/src/perturbench/configs/experiment/" + dataset_name + "/"
        os.makedirs(link_dir, exist_ok=True)
        link_path = os.path.join(link_dir, yaml_filename)

        # Remove existing symlink if it already exists
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)

        os.symlink(yaml_path, link_path)

        print(f"  Saved {csv_filename}, {h5ad_filename}, and {yaml_filename}")

@hydra.main(config_path="configs/splitter", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    covariate_keys = OmegaConf.to_container(cfg.covariates.names, resolve=True)

    main_dir = cfg.output.main_dir + cfg.dataset.name + '/'

    # Load data
    adata = sc.read_h5ad(cfg.adata.input_path)

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
    datasets = create_dataset_variants(adata, splitter, perturbations_to_remove, cfg.perturbations.key, covariate_keys[0], cfg.manual_control)

    # Save all datasets
    save_datasets(
        datasets,
        adata,
        cfg.dataset.name,
        cfg.perturbations.key,
        covariate_keys[0],
        cfg.perturbations.control_value,
        main_dir
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