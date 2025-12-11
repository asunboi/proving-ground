# storm/plugins/state/plugin.py
from pathlib import Path
import pandas as pd
from ..base import ModelPlugin
from importlib import resources
from jinja2 import Environment, FileSystemLoader
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from adjustText import adjust_text 
import pickle 
import math
import scanpy as sc
import pandas as pd
from scipy import sparse
import os

# module-level logger
log = logging.getLogger(__name__)

def _env():
    # templates packaged under storm/plugins/perturbench/templates/
    tmpl_dir = resources.files(__package__) / "templates"
    return Environment(loader=FileSystemLoader(str(tmpl_dir)),
                       autoescape=False, keep_trailing_newline=True,
                       trim_blocks=True, lstrip_blocks=True)

def _generate_from_template(template_name, **kwargs) -> str:
    return _env().get_template(template_name).render(**kwargs)

def _escape_key(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def _fmt_array(items):
    items = sorted({str(x) for x in items if pd.notna(x) and str(x).strip()})
    return "[ " + ", ".join(f'"{x}"' for x in items) + " ]"

def generate_toml_config(dataset_name, h5ad_path, df, perturbation_key, covariate_key,
                         split_col="transfer_split_seed1", train_label="train", control_value="ctrl") -> str:
    lines = []
    lines += [ "[datasets]", f'{dataset_name} = "{h5ad_path}"', "", "[training]", f'{dataset_name} = "{train_label}"', "", "[zeroshot]", "" ]
    ctrl = {control_value} if isinstance(control_value, str) else set(control_value)
    for cov, sub in df.groupby(covariate_key, sort=False, observed=False):
        val  = sub[(sub[split_col]=="val")  & (~sub[perturbation_key].isin(ctrl))][perturbation_key].tolist()
        test = sub[(sub[split_col]=="test") & (~sub[perturbation_key].isin(ctrl))][perturbation_key].tolist()
        if not val and not test:
            continue
        key = f'{dataset_name}.{cov}'
        lines += [f'[fewshot."{_escape_key(key)}"]', f"val = {_fmt_array(val)}", f"test = {_fmt_array(test)}", ""]
    return "\n".join(lines).rstrip()

def calculate_logfc_all(
    adata,
    control_value="NT_0",
    cell_type_col="predicted.subclass",
    condition_col="Assign",
    gene_name_col="gene_names",
    pseudocount=0.1,
    skip_missing=True,
):
    """
    Calculate log2 fold change vs control_value for *every* combination of
    (cell_type, perturbation) in adata.obs.

    Parameters
    ----------
    adata : AnnData
    control_value : str
        Label in `condition_col` to use as the reference (e.g. "NT_0").
    cell_type_col : str
        Column in adata.obs defining cell types (e.g. "predicted.subclass").
    condition_col : str
        Column in adata.obs defining perturbation / condition (e.g. "Assign").
    gene_name_col : str
        Column in adata.var containing gene names.
    pseudocount : float
        Pseudocount added before log2.
    skip_missing : bool
        If True, skip combinations where either perturbation or control
        has zero cells for that cell type. If False, include them as
        rows of NaNs.

    Returns
    -------
    logfc_all : pd.DataFrame
        Rows: MultiIndex (cell_type, perturbation)
        Columns: genes (from `gene_name_col`)
        Values: log2FC(perturbation vs control_value) within that cell type.
    """
    cell_types = adata.obs[cell_type_col].unique()
    conditions = adata.obs[condition_col].unique()

    rows = []
    indices = []

    for ct in cell_types:
        for cond in conditions:
            if cond == control_value:
                continue  # don't compare control vs itself

            # compute for this (cell_type, perturbation)
            X = adata.X
            obs = adata.obs

            mask_p = (obs[condition_col] == cond) & (obs[cell_type_col] == ct)
            mask_ctrl = (obs[condition_col] == control_value) & (
                obs[cell_type_col] == ct
            )

            if mask_p.sum() == 0 or mask_ctrl.sum() == 0:
                if skip_missing:
                    continue
                else:
                    # keep row with NaNs
                    n_genes = adata.n_vars
                    rows.append(np.full(n_genes, np.nan))
                    indices.append((ct, cond))
                    continue

            avg_p = X[mask_p].mean(axis=0)
            avg_ctrl = X[mask_ctrl].mean(axis=0)

            avg_p = avg_p.A1 if sparse.issparse(avg_p) else np.asarray(avg_p).ravel()
            avg_ctrl = (
                avg_ctrl.A1 if sparse.issparse(avg_ctrl) else np.asarray(avg_ctrl).ravel()
            )

            avg_p_log = np.log2(avg_p + pseudocount)
            avg_ctrl_log = np.log2(avg_ctrl + pseudocount)
            logfc = avg_p_log - avg_ctrl_log

            rows.append(logfc)

            ## SWAP FOR ONE INDEX SIMILAR TO PKL OUTPUT
            #indices.append((ct, cond))
            indices.append(f"{ct}_{cond}")

    if not rows:
        raise ValueError("No valid (cell_type, perturbation) combinations found.")

    gene_names = adata.var[gene_name_col].to_list()

    ## SWAP FOR ONE INDEX SIMILAR TO PKL OUTPUT
    #index = pd.MultiIndex.from_tuples(indices, names=[cell_type_col, condition_col])
    index = pd.Index(indices)

    logfc_all = pd.DataFrame(rows, index=index, columns=gene_names)
    return logfc_all

class Plugin(ModelPlugin):
    key = "state"

    def prepare_dirs(self, layout):
        pass

    def emit_for_split(self, df, dataset_name, split_name, h5ad_path,
                       perturbation_key, covariate_key, control_value, seed, layout):
        
        seed_dir = layout.seed_dir(seed)
        out_dir = layout.config_dir(seed_dir, f"state")
        
        toml_text = generate_toml_config(dataset_name, str(h5ad_path), df,
                                         perturbation_key, covariate_key,
                                         split_col=f"transfer_split_seed{seed}",
                                         train_label="train",
                                         control_value=control_value)
        
        toml_path = out_dir / f"{dataset_name}_{split_name}.toml"
        toml_path.write_text(toml_text)

        # write sbatch files
        sbatch_filename = f"{dataset_name}_{split_name}_seed{seed}.sbatch"
        state_dir = f"{dataset_name}_seed{seed}"
        sbatch = _generate_from_template(
            "state_sbatch.j2",
            toml_config_path=toml_path,
            perturbation_key=perturbation_key,
            covariate_key=covariate_key,
            control_value=control_value,
            dataset_name=dataset_name,
            output_dir=out_dir,
            state_dir=state_dir,
        )
        sbatch_path = out_dir / sbatch_filename
        sbatch_path.write_text(sbatch)

    def visualize_scatterplots(seeds):
        model_name = "STATE"
        
        # STATE latent 2k HVG
        boli = sc.read_h5ad(
            "/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/2025-11-21_12-06-21/data/test_full.h5ad"
        )

        # Directories, each containing adata_pred.h5ad and adata_real.h5ad
        adata_dirs = [
            f"/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli/seed_{seed}/configs/state/boli_seed{seed}"
            for seed in seeds
        ]
        
        genes = boli.var["gene_name"].values

        def load_deg_genes(cell_pert: str):
            """
            Load DEG genes for a specific cell type and perturbation.
            Modify the path pattern to match your file naming convention.
            """
            parts = cell_pert.split("_")
            cell_type = "_".join(parts[:-2])   # e.g., "L6 CT CTX"
            perturbation = parts[-2]           # e.g., "Tbr1"
            
            deg_file = (
                f"/gpfs/home/asun/jin_lab/perturbench/data/sig_deg/"
                f"{cell_type.replace(' ', '_')}_Assign{perturbation}_0.csv"
            )
            
            # try:
            #     deg_df = pd.read_csv(deg_file)
            #     return set(deg_df["Gene"].tolist())
            try:
                deg_df = pd.read_csv(deg_file)
                filtered = deg_df.loc[deg_df["adj_pval"] < 0.05, "Gene"]
                return set(filtered.dropna().astype(str).tolist())
            except FileNotFoundError:
                print(f"Warning: DEG file not found for {cell_pert}: {deg_file}")
                return set()

        for run_dir in adata_dirs:
            adata_pred_path = os.path.join(run_dir, "eval_final.ckpt/adata_pred.h5ad")
            adata_real_path = os.path.join(run_dir, "eval_final.ckpt/adata_real.h5ad")

            if not (os.path.exists(adata_pred_path) and os.path.exists(adata_real_path)):
                print(f"Skipping {run_dir}: missing adata_pred.h5ad or adata_real.h5ad")
                continue

            print(f"Processing {run_dir}")

            adata_pred = sc.read_h5ad(adata_pred_path)
            adata_real = sc.read_h5ad(adata_real_path)

            logfc_all_real = calculate_logfc_all(adata_real)
            logfc_all_pred = calculate_logfc_all(adata_pred)

            df_pred = logfc_all_pred  # rows: cell_pert, cols: genes
            df_ref  = logfc_all_real

            cell_perts = df_pred.index.tolist()

            # Make a "figures" subdirectory next to the ckpt
            fig_dir = os.path.join(run_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)

            # Plot for each cell_pert individually
            for cell_pert in cell_perts:

                # Plot 1: All genes
                fig, ax = plt.subplots(figsize=(8, 8))
                
                x_vals = df_ref.loc[cell_pert].values
                y_vals = df_pred.loc[cell_pert].values
                
                pearson_r, p_value = pearsonr(x_vals, y_vals)

                top10_idx = np.argsort(x_vals)[-10:]
                bottom10_idx = np.argsort(x_vals)[:10]
                top_and_bottom_idx = set(top10_idx).union(set(bottom10_idx))

                ax.scatter(x_vals, y_vals, alpha=0.7, color="lightgray")
                ax.plot([x_vals.min(), x_vals.max()], [x_vals.min(), x_vals.max()], 'k--', alpha=0.5)
                ax.set_xlabel("Actual change in expression")
                ax.set_ylabel(f"Predicted expression ({model_name})")
                ax.set_title(f"{model_name} Prediction for {cell_pert}\nPearson r = {pearson_r:.3f} (p = {p_value:.2e})")
                ax.grid(True)

                texts = []
                for idx in top_and_bottom_idx:
                    color = "steelblue" if idx in top10_idx else "firebrick"
                    texts.append(
                        ax.text(
                            x_vals[idx], y_vals[idx], genes[idx],
                            fontsize=12,
                            color=color,
                            alpha=0.9,
                            ha='right' if idx in top10_idx else 'left',
                            va='bottom'
                        )
                    )
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), ax=ax)
            

                # Example: saving a figure named "my_plot"
                fig_name = f"{cell_pert}_all_genes.png"
                fig_path = os.path.join(fig_dir, fig_name)

                plt.tight_layout()
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                
                # Plot 2: DEGs only for this specific cell_pert
                deg_genes = load_deg_genes(cell_pert)
                
                if len(deg_genes) == 0:
                    # No DEG list available for this cell_pert
                    print(f"No DEGs found for {cell_pert}, skipping DEG correlation")
                    continue

                # Intersect DEGs with the genes present in the logFC matrices
                common_deg_genes = list(set(deg_genes) & set(df_pred.columns))
                if len(common_deg_genes) == 0:
                    print(f"DEGs for {cell_pert} not found in logFC genes, skipping")
                    continue

                n_deg = len(common_deg_genes)

                if n_deg < 2:
                    print(f"Not enough DEGs for {cell_pert} (n={n_deg}), skipping DEG correlation")
                    continue
                
                if len(deg_genes) > 0:
                    # Create mask for DEG genes
                    deg_mask = np.array([gene in deg_genes for gene in genes])
                    deg_indices = np.where(deg_mask)[0]
                    
                    # Subset dataframes to only DEG genes
                    df_pred_deg = df_pred.iloc[:, deg_indices]
                    df_ref_deg = df_ref.iloc[:, deg_indices]
                    genes_deg = genes[deg_mask]
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    x_vals_deg = df_ref_deg.loc[cell_pert].values
                    y_vals_deg = df_pred_deg.loc[cell_pert].values

                    pearson_r, p_value = pearsonr(x_vals_deg, y_vals_deg)

                    top10_idx = np.argsort(x_vals_deg)[-10:]
                    bottom10_idx = np.argsort(x_vals_deg)[:10]
                    top_and_bottom_idx = set(top10_idx).union(set(bottom10_idx))

                    # Plot all DEG genes in green
                    ax.scatter(x_vals_deg, y_vals_deg, alpha=0.7, color="green", s=20)

                    ax.plot([x_vals_deg.min(), x_vals_deg.max()], [x_vals_deg.min(), x_vals_deg.max()], 'k--', alpha=0.5)
                    ax.set_xlabel("Actual change in expression")
                    ax.set_ylabel(f"Predicted expression ({model_name})")
                    ax.set_title(f"{model_name} Prediction for {cell_pert}\nPearson r = {pearson_r:.3f} (p = {p_value:.2e})\nDEGs only (n={len(genes_deg)})")
                    ax.grid(True)
                    
                    # texts = []
                    # for idx in top_and_bottom_idx:
                    #     color = "steelblue" if idx in top10_idx else "firebrick"
                    #     texts.append(
                    #         ax.text(
                    #             x_vals_deg[idx], y_vals_deg[idx], genes_deg[idx],
                    #             fontsize=12,
                    #             color=color,
                    #             alpha=0.9,
                    #             ha='right' if idx in top10_idx else 'left',
                    #             va='bottom'
                    #         )
                    #     )
                    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), ax=ax)
                    
                    plt.tight_layout()
                    # Example: saving a figure named "my_plot"
                    deg_dir = os.path.join(fig_dir, "deg")
                    os.makedirs(deg_dir, exist_ok=True)
                    fig_name = f"{cell_pert}_DEGs_only.png"
                    fig_path = os.path.join(deg_dir, fig_name)
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    print(f"Completed plots for {cell_pert}: {len(genes)} total genes, {len(genes_deg)} DEGs")
                else:
                    print(f"No DEGs found for {cell_pert}, skipping DEG plot")