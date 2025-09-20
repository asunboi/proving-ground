import scanpy as sc
from perturbench.data.datasplitter import PerturbationDataSplitter
import matplotlib.pyplot as plt
import numpy as np
import yaml
from omegaconf import OmegaConf
import pandas as pd

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


def plot_counts(df, cell_line_col="cell_line", gene_col="gene", seed_col="transfer_split_seed1"):
    """
    Make heatmaps of counts per (cell_line, gene) × seed from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least [cell_line_col, gene_col, seed_col].
    cell_line_col : str, default "cell_line"
        Column containing cell line identifiers.
    gene_col : str, default "gene"
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
        df.groupby([gene_col, seed_col, cell_line_col])
        .size()
        .reset_index(name="count")
    )

    # pivot to wide format
    cell_counts_all = count_long.pivot(
        index=[cell_line_col, gene_col],
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

    cell_classes = filtered[cell_line_col].unique()
    n_cols = 2
    n_rows = (len(cell_classes) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, cell_class in enumerate(cell_classes):
        ax = axes[i]
        subset = (
            filtered[filtered[cell_line_col] == cell_class]
            .set_index(gene_col)
            .drop(columns=cell_line_col)
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
    plt.show()
    return fig, axes

boli = sc.read_h5ad('/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_ctx_scprocess_no+ctrl.h5ad')

df_100 = boli

perturbations_to_remove = ["XPO7", "RB1CC1", "SATB2", "CX3CL1", "CUL1", "TBR1"]
df_100_5 = boli.obs[~boli.obs["condition"].isin(perturbations_to_remove)]

balanced_transfer_splitter = PerturbationDataSplitter(
    df_100_5,
    perturbation_key='condition',
    covariate_keys=['cell_class'],
    perturbation_control_value='ctrl',
)

seed_id = 1
balanced_transfer_split = balanced_transfer_splitter.split_covariates(
    seed=seed_id, 
    print_split=True, 
    #max_heldout_covariates=3, ## Maximum number of held out covariates (in this case cell types)
    max_heldout_fraction_per_covariate=0.6, ## Maximum fraction of perturbations held out per covariate
    max_heldout_covariates = 3,
    #test_fraction = 0.5,
)

# Define which perturbations to add back
perturbations_to_include = ["CX3CL1", "CUL1", "TBR1"]

# Get the rows from df_100 that match perturbations_to_include
rows_to_add = df_100[df_100["condition"].isin(perturbations_to_include)].copy()

# Set transfer_split_seed1 to "train" for these rows
rows_to_add["transfer_split_seed1"] = "train"

# Combine df_100_5 with the modified rows
df_100_8 = pd.concat([df_100_5, rows_to_add], ignore_index=False)

# Define which perturbations to add back
perturbations_to_include = ["XPO7", "RB1CC1", "SATB2"]

# Get the rows from df_100 that match perturbations_to_include
rows_to_add = df_100[df_100["condition"].isin(perturbations_to_include)].copy()

# Set transfer_split_seed1 to "train" for these rows
rows_to_add["transfer_split_seed1"] = "train"

# Combine df_100_5 with the modified rows
df_100_11 = pd.concat([df_100_8, rows_to_add], ignore_index=False)

# Example usage
group_cols = ["condition", "cell_class", "transfer_split_seed1"]

df_60_11 = stratified_subsample_train(df_100_11, frac=0.6, group_keys=group_cols, random_state=1)
df_60_8 = stratified_subsample_train(df_100_8, frac=0.6, group_keys=group_cols, random_state=1)
df_60_5 = stratified_subsample_train(df_100_5, frac=0.6, group_keys=group_cols, random_state=1)

df_20_11 = stratified_subsample_train(df_100_11, frac=0.2, group_keys=group_cols, random_state=2)
df_20_8 = stratified_subsample_train(df_100_8, frac=0.2, group_keys=group_cols, random_state=2)
df_20_5 = stratified_subsample_train(df_100_5, frac=0.2, group_keys=group_cols, random_state=2)

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_60[["transfer_split_seed1"]].copy()

# write both the index and the column
out.to_csv("boli_df_60_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_20[["transfer_split_seed1"]].copy()

# write both the index and the column
out.to_csv("boli_df_20_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

adata_60 = boli[df_60.index].copy()
adata_20 = boli[df_20.index].copy()
adata_60.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_60.h5ad")
adata_20.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_20.h5ad")

df_100 = balanced_transfer_splitter.obs_dataframe

# Suppose you have a list of perturbations to remove
perturbations_to_remove = ["XPO7", "RB1CC1", "SATB2"]

# Drop any row whose condition is in that list
df_100_8 = df_100[~df_100["condition"].isin(perturbations_to_remove)]
df_60_8 = df_60[~df_60["condition"].isin(perturbations_to_remove)]
df_20_8 = df_20[~df_20["condition"].isin(perturbations_to_remove)]

# Suppose you have a list of perturbations to remove
perturbations_to_remove = ["XPO7", "RB1CC1", "SATB2", "CX3CL1", "CUL1", "TBR1"]

# Drop any row whose condition is in that list
df_100_5 = df_100[~df_100["condition"].isin(perturbations_to_remove)]
df_60_5 = df_60[~df_60["condition"].isin(perturbations_to_remove)]
df_20_5 = df_20[~df_20["condition"].isin(perturbations_to_remove)]

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_100_8[["transfer_split_seed1"]].copy()
# write both the index and the column
out.to_csv("boli_df_100_8_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_60_8[["transfer_split_seed1"]].copy()
# write both the index and the column
out.to_csv("boli_df_60_8_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_20_8[["transfer_split_seed1"]].copy()
# write both the index and the column
out.to_csv("boli_df_20_8_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_100_5[["transfer_split_seed1"]].copy()
# write both the index and the column
out.to_csv("boli_df_100_5_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_60_5[["transfer_split_seed1"]].copy()
# write both the index and the column
out.to_csv("boli_df_60_5_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

# df_60 is your subsampled dataframe with cell barcodes as index
out = df_20_5[["transfer_split_seed1"]].copy()
# write both the index and the column
out.to_csv("boli_df_20_5_train_downsample_only_split.csv", index=True, index_label="cell_barcode", header=False)

adata_100_8 = boli[df_100_8.index].copy()
adata_100_5 = boli[df_100_5.index].copy()
adata_100_8.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_100_8.h5ad")
adata_100_5.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_100_5.h5ad")

adata_60_8 = boli[df_60_8.index].copy()
adata_60_5 = boli[df_60_5.index].copy()
adata_60_8.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_60_8.h5ad")
adata_60_5.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_60_5.h5ad")

adata_20_8 = boli[df_20_8.index].copy()
adata_20_5 = boli[df_20_5.index].copy()
adata_20_8.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_20_8.h5ad")
adata_20_5.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/boli_subset_seed2_train_downsample_only_20_5.h5ad")


# ## visualization?
# # count_long = balanced_transfer_splitter.obs_dataframe.groupby(["condition", "transfer_split_seed1", "cell_class"]).size().reset_index(name = "count")

# # cell_counts_all = count_long.pivot(index = ["cell_class", "condition"], columns = "transfer_split_seed1", values = "count" ).reset_index()

# # cell_counts_all.columns.name = ""
# # filtered = cell_counts_all.dropna(axis=0, how="all", subset=cell_counts_all.columns[2:])
# # cell_classes = filtered["cell_class"].unique()
# # n_cols = 3
# # n_rows = (len(cell_classes) + n_cols - 1) // n_cols

# # fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
# # axes = axes.flatten()

# # for i, cell_class in enumerate(cell_classes):
# #     ax = axes[i]
# #     subset = filtered[filtered["cell_class"] == cell_class].set_index("condition")
# #     subset = subset.drop(columns="cell_class")
# #     subset = subset.fillna(0)
# #     data = subset.values

# #     im = ax.imshow(data, aspect="auto", cmap="viridis")
# #     ax.set_title(cell_class)
# #     ax.set_xticks(np.arange(subset.shape[1]))
# #     ax.set_xticklabels(subset.columns, rotation=45, ha="right")
# #     ax.set_yticks(np.arange(subset.shape[0]))
# #     ax.set_yticklabels(subset.index)

# #     for y in range(data.shape[0]):
# #         for x in range(data.shape[1]):
# #             value = data[y, x]
# #             if not np.isnan(value):
# #                 ax.text(x, y, int(value), ha="center", va="center", color="white" if value < data.max() / 2 else "black")

# #     fig.colorbar(im, ax=ax, shrink=0.6)

# # for j in range(i + 1, len(axes)):
# #     fig.delaxes(axes[j])

# # plt.tight_layout()
# # plt.show()