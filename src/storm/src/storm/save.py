from dataclasses import dataclass
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from storm.plugins.loader import load_plugins

log = logging.getLogger(__name__)

def plot_counts(df, outfile, seed, cfg):
    """
    Make heatmaps of counts per (batch, cell_line, gene) × seed from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least [cell_line_col, gene_col, seed_col].
        Optionally [batch_col] if cfg.data.batch_key is not None.
    cfg.data.batch_key : str or None
        Column containing batch identifiers. If None, batch grouping is skipped.
    cfg.data.covariate_key : str, default "cell_line"
        Column containing cell line identifiers.
    cfg.data.perturbation_key : str, default "gene"
        Column containing gene identifiers.
    seed_col : str, default "transfer_split_seed1"
        Column containing split/seed values.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes
    """
    seed_col = "transfer_split_seed" + str(seed)
    
    # Determine whether to include batch_key
    has_batch = cfg.data.batch_key is not None
    
    # Build groupby columns dynamically
    groupby_cols = [cfg.data.perturbation_key, seed_col]
    if has_batch:
        groupby_cols.extend([cfg.data.batch_key, cfg.data.covariate_key])
    else:
        groupby_cols.append(cfg.data.covariate_key)

    # counts
    count_long = (
        df.groupby(groupby_cols, observed=False)
        .size()
        .reset_index(name="count")
    )

    # Build pivot index dynamically
    pivot_index = [cfg.data.batch_key, cfg.data.covariate_key, cfg.data.perturbation_key] if has_batch else [cfg.data.covariate_key, cfg.data.perturbation_key]
    
    # pivot to wide format
    cell_counts_all = count_long.pivot(
        index=pivot_index,
        columns=seed_col,
        values="count"
    ).reset_index()

    cell_counts_all.columns.name = ""

    # drop rows with all NaN counts
    num_index_cols = 3 if has_batch else 2
    filtered = cell_counts_all.dropna(
        axis=0,
        how="all",
        subset=cell_counts_all.columns[num_index_cols:]
    )

    # Get unique combinations
    if has_batch:
        combinations = filtered[[cfg.data.batch_key, cfg.data.covariate_key]].drop_duplicates()
    else:
        combinations = filtered[[cfg.data.covariate_key]].drop_duplicates()
    
    n_cols = 2
    n_rows = (len(combinations) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(combinations.iterrows()):
        cell_class = row[cfg.data.covariate_key]
        
        ax = axes[i]
        
        # Build filter condition and subset
        if has_batch:
            batch_val = row[cfg.data.batch_key]
            mask = (filtered[cfg.data.batch_key] == batch_val) & (filtered[cfg.data.covariate_key] == cell_class)
            cols_to_drop = [cfg.data.batch_key, cfg.data.covariate_key]
            title = f"{batch_val} - {cell_class}"
        else:
            mask = filtered[cfg.data.covariate_key] == cell_class
            cols_to_drop = [cfg.data.covariate_key]
            title = str(cell_class)
        
        subset = (
            filtered[mask]
            .set_index(cfg.data.perturbation_key)
            .drop(columns=cols_to_drop)
            .fillna(0)
        )

        # Reorder rows so that control (NT_0) is always at the bottom
        idx = list(subset.index)
        if cfg.data.control_value in idx:
            idx = [x for x in idx if x != cfg.data.control_value] + [cfg.data.control_value]
            subset = subset.loc[idx]

        data = subset.values

        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_title(title)
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

def plot_counts_rasterized(df, outfile, seed, cfg):
    import numpy as np
    import matplotlib.pyplot as plt

    seed_col = f"transfer_split_seed{seed}"
    has_batch = cfg.data.batch_key is not None

    groupby_cols = [cfg.data.perturbation_key, seed_col]
    if has_batch:
        groupby_cols.extend([cfg.data.batch_key, cfg.data.covariate_key])
    else:
        groupby_cols.append(cfg.data.covariate_key)

    count_long = (
        df.groupby(groupby_cols, observed=False)
        .size()
        .reset_index(name="count")
    )

    pivot_index = (
        [cfg.data.batch_key, cfg.data.covariate_key, cfg.data.perturbation_key]
        if has_batch
        else [cfg.data.covariate_key, cfg.data.perturbation_key]
    )

    wide = (
        count_long
        .pivot(index=pivot_index, columns=seed_col, values="count")
        .fillna(0)
        .reset_index()
    )

    index_cols = 3 if has_batch else 2
    data_cols = wide.columns[index_cols:]
    keep_data_cols = wide[data_cols].notna().any(axis=0)

    wide = wide.loc[:, list(wide.columns[:index_cols]) + list(data_cols[keep_data_cols])]

    if has_batch:
        groups = wide[[cfg.data.batch_key, cfg.data.covariate_key]].drop_duplicates()
    else:
        groups = wide[[cfg.data.covariate_key]].drop_duplicates()

    n_cols = 2
    n_rows = (len(groups) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(groups.iterrows()):
        ax = axes[i]

        if has_batch:
            mask = (
                (wide[cfg.data.batch_key] == row[cfg.data.batch_key]) &
                (wide[cfg.data.covariate_key] == row[cfg.data.covariate_key])
            )
            title = f"{row[cfg.data.batch_key]} – {row[cfg.data.covariate_key]}"
            drop_cols = [cfg.data.batch_key, cfg.data.covariate_key]
        else:
            mask = wide[cfg.data.covariate_key] == row[cfg.data.covariate_key]
            title = str(row[cfg.data.covariate_key])
            drop_cols = [cfg.data.covariate_key]

        subset = (
            wide[mask]
            .set_index(cfg.data.perturbation_key)
            .drop(columns=drop_cols)
        )

        # ---- IMPORTANT PART ----
        # sort perturbations by variability across seeds
        order = subset.std(axis=1).sort_values(ascending=False).index
        subset = subset.loc[order]

        data = subset.values

        # tweak since control can dominate
        data = np.log1p(data)

        im = ax.imshow(
            data,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            rasterized=True
        )

        ax.set_title(title)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Perturbations (sorted)")
        ax.set_xticks(np.arange(len(subset.columns)))
        ax.set_xticklabels(subset.columns, rotation=45, ha="right")

        # critical: remove y tick labels
        ax.set_yticks([])

        fig.colorbar(im, ax=ax, shrink=0.6)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)

    return fig

# helper to count perturbations to determine which plotting mechanism to use
def _n_perturbations(df, cfg):
    return df[cfg.data.perturbation_key].nunique()

def plot_counts_auto(
    df,
    outfile,
    seed,
    cfg,
    max_annotated=20,
):
    """
    Automatically choose plotting strategy based on number of perturbations.

    - <= max_annotated: annotated heatmap (plot_counts)
    - >  max_annotated: rasterized heatmap (plot_counts_rasterized)
    """

    n_perts = _n_perturbations(df, cfg)

    if n_perts <= max_annotated:
        plot_counts(df, outfile, seed, cfg)
    else:
        plot_counts_rasterized(df, outfile, seed, cfg)

# projectlayout defines the shared folders with __post_init__, while emit_for_split in the plugins call config_dir to create their respective packages.
@dataclass
class ProjectLayout:
    main_dir: Path
    dataset_name: str

    def __post_init__(self):
        self.data_dir   = self.main_dir / "data"
        #self.splits_dir = self.main_dir / "splits"
        self.fig_dir    = self.main_dir / "figures"
        for d in (self.data_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)

    def config_dir(self, seed_dir, model_key: str) -> Path:
        """Return (and create) a configs/<model_key> directory."""
        p = seed_dir / "configs" / model_key
        p.mkdir(parents=True, exist_ok=True)
        return p

    def seed_dir(self, seed: int) -> Path:
        """
        Return (and create) the directory for a given seed:
        <main_dir>/seed_{seed}
        """
        d = self.main_dir / f"seed_{seed}" 
        d.mkdir(parents=True, exist_ok=True)
        return d

    def prediction_path(self) -> Path:
        return self.main_dir / "prediction_dataframe.csv"
    
def save_datasets(datasets, adata, cfg):
    layout = ProjectLayout(Path(cfg.output.main_dir), cfg.data.name)
    plugins = load_plugins(cfg.models)   # imports only the plugins you asked for

    for p in plugins:
        p.prepare_dirs(layout)

    for split_name, df in datasets.items():
        # shared artifacts
        #csv_path = layout.splits_dir / f"{dataset_name}_{split_name}_split.csv"
        #df[["transfer_split_seed1"]].to_csv(csv_path, index=True, index_label="cell_barcode", header=False)

        h5ad_path = layout.data_dir / f"{cfg.data.name}_{split_name}.h5ad"
        adata[df.index].write_h5ad(str(h5ad_path))

        for seed in cfg.splitter.seed:
            # create seed directory
            seed_dir = layout.seed_dir(seed)

            # create figures of seeded splits
            fig_path = layout.fig_dir / f"{cfg.data.name}_{split_name}_seed{seed}.png"
            plot_counts_auto(df, str(fig_path), seed, cfg)

            # per-model artifacts
            for plugin in plugins:
                plugin.emit_for_split(df, 
                                      split_name,
                                      h5ad_path, 
                                      seed, 
                                      layout,
                                      cfg)
