import scanpy as sc
from perturbench.data.datasplitter import PerturbationDataSplitter
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
import os
import hydra
from omegaconf import DictConfig, OmegaConf

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


def create_dataset_variants(adata, balanced_transfer_splitter, perturbations, perturbation_key, covariate_key, base_fractions=[1.0, 0.6, 0.2]):
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
        split_value_to_evaluate: val
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


def save_datasets(datasets, adata, dataset_name, perturbation_key, covariate_key, control_value, output_dir="/gpfs/home/asun/jin_lab/perturbench/0_datasets/clean/", 
                 csv_dir="/gpfs/home/asun/jin_lab/perturbench/1_train/", fig_dir="/gpfs/home/asun/jin_lab/perturbench/1_train/", yaml_dir=None):
    """
    Saves all datasets as CSV files and H5AD files, and generates corresponding YAML config files
    
    Parameters:
    - datasets: Dictionary of dataset name to dataframe
    - adata: The original AnnData object 
    - output_dir: Directory to save H5AD files
    - csv_dir: Directory to save CSV files
    - yaml_dir: Directory to save YAML files (defaults to csv_dir if None)
    """

    if yaml_dir is None:
        yaml_dir = csv_dir
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(yaml_dir, exist_ok=True)
    
    for split_name, df in datasets.items():
        print(f"Saving {split_name}...")
        
        # Save CSV with transfer split information
        out = df[["transfer_split_seed1"]].copy()
        csv_filename = f"{dataset_name}_{split_name}_split.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        out.to_csv(csv_path, index=True, index_label="cell_barcode", header=False)
        
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
        
        print(f"  Saved {csv_filename}, {h5ad_filename}, and {yaml_filename}")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    covariate_keys = OmegaConf.to_container(cfg.covariates.names, resolve=True)

    # Load data
    adata = sc.read_h5ad(cfg.adata.input_path)

    # Create initial filtered dataset
    perturbations_to_remove = OmegaConf.to_container(cfg.perturbations.remove, resolve=True)
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
    datasets = create_dataset_variants(adata, splitter, perturbations_to_remove, cfg.perturbations.key, covariate_keys[0])

    # Save all datasets
    save_datasets(
        datasets,
        adata,
        cfg.dataset.name,
        cfg.perturbations.key,
        covariate_keys[0],
        cfg.perturbations.control_value,
        output_dir=cfg.output.data_dir,
        csv_dir=cfg.output.csv_dir,
        fig_dir=cfg.output.fig_dir,
        yaml_dir=cfg.output.yaml_dir,
    )

    print("All datasets created and saved successfully!")

# Main execution
if __name__ == "__main__":
    main()