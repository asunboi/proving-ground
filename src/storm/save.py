from dataclasses import dataclass
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import logging
from plugins.loader import load_plugins

log = logging.getLogger(__name__)

# jinja2 setup
# configure template environment once at module level
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=False,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Adjust once, reuse everywhere
EXPERIMENT_CONFIG_ROOT = Path(
    "/gpfs/home/asun/jin_lab/perturbench/src/perturbench/src/perturbench/configs/experiment"
)

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
        df.groupby([perturbation_key, seed_col, covariate_key], observed=False)
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

        for cov_value, sub in df.groupby(covariate_key, sort=False, observed=False):
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

def generate_yaml_config(model_name, 
                         dataset_name, 
                         split_name, 
                         h5ad_path, 
                         perturbation_key, 
                         covariate_key, 
                         control_value, 
                         csv_path, 
                         template_name="boli_ctx.yaml.j2"):
    """
    Render a YAML config from Jinja2 template.
    
    Parameters:
    - split_name: Name of the dataset (e.g., "100_8")
    - h5ad_path: Full path to the H5AD file
    - csv_path: Full path to the CSV split file
    
    Returns:
    - yaml_content: String containing the YAML configuration
    """
    template = env.get_template(template_name)
    yaml_content = template.render(
            model_name=model_name,
            dataset_name=dataset_name,
            split_name=split_name,
            h5ad_path=h5ad_path,
            perturbation_key=perturbation_key,
            covariate_key=covariate_key,
            control_value=control_value,
            csv_path=csv_path,
        )
    return yaml_content

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
    
    log.info(f"Extracted {len(test_combos)} unique test combinations")
    log.info(f"Saved to: {output_csv}")
    
    return test_combos

@dataclass
class SplitLayout:
    main_dir: Path
    dataset_name: str

    def __post_init__(self):
        self.data_dir    = self.main_dir / "data"
        self.csv_dir     = self.main_dir / "splits"
        self.fig_dir     = self.main_dir / "figures"
        self.linear_dir  = self.main_dir / "linear"
        self.latent_dir  = self.main_dir / "latent"
        self.toml_dir    = self.main_dir / "toml"
        # later: self.scripts_dir = self.main_dir / "scripts"
        #        self.outputs_dir = self.main_dir / "outputs"

        for d in (
            self.data_dir,
            self.csv_dir,
            self.fig_dir,
            self.linear_dir,
            self.latent_dir,
            self.toml_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # where symlinks go for this dataset
        self.experiment_link_dir = EXPERIMENT_CONFIG_ROOT / self.dataset_name
        self.experiment_link_dir.mkdir(parents=True, exist_ok=True)

    def prediction_path(self) -> Path:
        # if you really want one shared file per dataset
        return self.main_dir / "prediction_dataframe.csv"

## REFACTOR: working on plugin implementation
# def save_datasets(
#     datasets,
#     adata,
#     dataset_name: str,
#     perturbation_key: str,
#     covariate_key: str,
#     control_value: str,
#     main_dir: str | Path,
# ):
#     """
#     High-level orchestration: loop over splits and delegate to helpers.
#     """
#     layout = SplitLayout(Path(main_dir), dataset_name)

#     for split_name, df in datasets.items():
#         log.info(f"Saving {split_name}...")
#         save_single_split(
#             df=df,
#             split_name=split_name,
#             adata=adata,
#             layout=layout,
#             dataset_name=dataset_name,
#             perturbation_key=perturbation_key,
#             covariate_key=covariate_key,
#             control_value=control_value,
#         )

# projectlayout defines the shared folders with __post_init__, while emit_for_split in the plugins call config_dir to create their respective packages.
@dataclass
class ProjectLayout:
    main_dir: Path
    dataset_name: str

    def __post_init__(self):
        self.data_dir   = self.main_dir / "data"
        self.splits_dir = self.main_dir / "splits"
        self.fig_dir    = self.main_dir / "figures"
        for d in (self.data_dir, self.splits_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)

    def config_dir(self, model_key: str) -> Path:
        """Return (and create) a configs/<model_key> directory."""
        p = self.main_dir / "configs" / model_key
        p.mkdir(parents=True, exist_ok=True)
        return p

    def prediction_path(self) -> Path:
        return self.main_dir / "prediction_dataframe.csv"
    
def save_datasets(datasets, adata, dataset_name, perturbation_key, covariate_key,
                  control_value, main_dir, models):
    layout = ProjectLayout(Path(main_dir), dataset_name)
    plugins = load_plugins(models)   # imports only the plugins you asked for

    for p in plugins:
        p.prepare_dirs(layout)

    for split_name, df in datasets.items():
        # shared artifacts
        csv_path = layout.splits_dir / f"{dataset_name}_{split_name}_split.csv"
        df[["transfer_split_seed1"]].to_csv(csv_path, index=True, index_label="cell_barcode", header=False)

        fig_path = layout.fig_dir / f"{dataset_name}_{split_name}_split.png"
        plot_counts(df, str(fig_path), perturbation_key, covariate_key)

        h5ad_path = layout.data_dir / f"{dataset_name}_{split_name}.h5ad"
        adata[df.index].write_h5ad(str(h5ad_path))

        # per-model artifacts
        for plugin in plugins:
            plugin.emit_for_split(df, dataset_name, split_name, h5ad_path, csv_path,
                                  perturbation_key, covariate_key, control_value, layout)

def save_single_split(
    df,
    split_name: str,
    adata,
    layout: SplitLayout,
    dataset_name: str,
    perturbation_key: str,
    covariate_key: str,
    control_value: str,
):
    # 1) CSV with transfer_split_seed1
    csv_path = save_split_csv(df, dataset_name, split_name, layout.csv_dir)

    # 2) Prediction dataframe (per split or shared; here one per dataset)
    prediction_path = layout.prediction_path()
    generate_prediction_dataframe(
        df,
        condition_col=perturbation_key,
        cell_type_col=covariate_key,
        output_csv=str(prediction_path),
    )

    # 3) Plot split
    fig_path = layout.fig_dir / f"{dataset_name}_{split_name}_split.png"
    plot_counts(df, str(fig_path), perturbation_key, covariate_key)

    # 4) H5AD subset
    h5ad_path = save_split_h5ad(
        adata=adata,
        df=df,
        dataset_name=dataset_name,
        split_name=split_name,
        data_dir=layout.data_dir,
    )

    # 5) YAMLs (linear + latent) + symlinks
    save_model_yaml_configs(
        dataset_name=dataset_name,
        split_name=split_name,
        h5ad_path=h5ad_path,
        perturbation_key=perturbation_key,
        covariate_key=covariate_key,
        control_value=control_value,
        csv_path=csv_path,
        linear_dir=layout.linear_dir,
        latent_dir=layout.latent_dir,
        experiment_link_dir=layout.experiment_link_dir,
    )

    # 6) TOML
    save_toml_config(
        df=df,
        dataset_name=dataset_name,
        split_name=split_name,
        h5ad_path=h5ad_path,
        perturbation_key=perturbation_key,
        covariate_key=covariate_key,
        control_value=control_value,
        toml_dir=layout.toml_dir,
    )


def save_split_csv(df, dataset_name: str, split_name: str, csv_dir: Path) -> Path:
    csv_filename = f"{dataset_name}_{split_name}_split.csv"
    csv_path = csv_dir / csv_filename

    out = df[["transfer_split_seed1"]].copy()
    out.to_csv(csv_path, index=True, index_label="cell_barcode", header=False)

    return csv_path


def save_split_h5ad(adata, df, dataset_name: str, split_name: str, data_dir: Path) -> Path:
    adata_subset = adata[df.index].copy()
    h5ad_filename = f"{dataset_name}_{split_name}.h5ad"
    h5ad_path = data_dir / h5ad_filename
    adata_subset.write_h5ad(str(h5ad_path))
    return h5ad_path


def save_model_yaml_configs(
    dataset_name: str,
    split_name: str,
    h5ad_path: Path,
    perturbation_key: str,
    covariate_key: str,
    control_value: str,
    csv_path: Path,
    linear_dir: Path,
    latent_dir: Path,
    experiment_link_dir: Path,
):
    model_specs = {
        "linear_additive": linear_dir,
        "latent_additive": latent_dir,
    }

    for model_name, model_dir in model_specs.items():
        yaml_content = generate_yaml_config(
            model_name,
            dataset_name,
            split_name,
            str(h5ad_path),
            perturbation_key,
            covariate_key,
            control_value,
            str(csv_path),
        )

        yaml_filename = f"{dataset_name}_{split_name}.yaml"
        yaml_path = model_dir / yaml_filename
        yaml_path.write_text(yaml_content)

        # symlink into perturbench experiment configs
        link_path = experiment_link_dir / yaml_filename
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(yaml_path, link_path)


def save_toml_config(
    df,
    dataset_name: str,
    split_name: str,
    h5ad_path: Path,
    perturbation_key: str,
    covariate_key: str,
    control_value: str,
    toml_dir: Path,
):
    toml_content = generate_toml_config(
        dataset_name=dataset_name,
        h5ad_path=str(h5ad_path),
        df=df,
        perturbation_key=perturbation_key,
        covariate_key=covariate_key,
        split_col="transfer_split_seed1",
        train_label="train",
        control_value=control_value,
    )

    toml_filename = f"{dataset_name}_{split_name}.toml"
    toml_path = toml_dir / toml_filename
    toml_path.write_text(toml_content)