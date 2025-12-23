# storm/plugins/perturbench/plugin.py
from pathlib import Path
import os
from importlib import resources
from jinja2 import Environment, FileSystemLoader
from ..base import ModelPlugin
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
import seaborn as sns
import anndata as ad

# module-level logger
log = logging.getLogger(__name__)

def _env():
    # templates packaged under storm/plugins/perturbench/templates/
    tmpl_dir = resources.files(__package__) / "templates"
    return Environment(loader=FileSystemLoader(str(tmpl_dir)),
                       autoescape=False, keep_trailing_newline=True,
                       trim_blocks=True, lstrip_blocks=True)

def generate_yaml_config(template_name, **ctx) -> str:
    return _env().get_template(template_name).render(**ctx)

def calculate_logfc_all(
    adata,
    control_value="NT_0",
    cell_type_col="predicted.subclass",
    condition_col="Assign",
    gene_name_col="gene_names" ,
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

    # --- gene names ---
    use_var_names = (
        gene_name_col is None
        or str(gene_name_col).lower() in {"index", "var_names", "varnames"}
        or gene_name_col not in adata.var.columns
    )

    if use_var_names:
        gene_names = adata.var_names.to_list()
    else:
        gene_names = adata.var[gene_name_col].astype(str).to_list()

    ## SWAP FOR ONE INDEX SIMILAR TO PKL OUTPUT
    #index = pd.MultiIndex.from_tuples(indices, names=[cell_type_col, condition_col])
    index = pd.Index(indices)

    logfc_all = pd.DataFrame(rows, index=index, columns=gene_names)
    return logfc_all

def load_and_join_anndatas(
    dir_path: str,
    pattern: str = "*.h5ad",
    join: str = "outer",          # "outer" keeps union of genes; "inner" keeps intersection
    label: str = "source_file",   # adds obs column showing which file each cell came from
    index_unique: str = "-",      # makes obs_names unique across files
):
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {dir_path}")

    adatas = []
    for fp in files:
        a = sc.read_h5ad(fp)              # loads fully into memory
        a.obs[label] = fp.name
        adatas.append(a)

    # Concatenate along cells (obs)
    adata_merged = ad.concat(
        adatas,
        axis=0,
        join=join,
        label=label,            # also creates a categorical "source_file" (same as we set)
        keys=[f.name for f in files],
        index_unique=index_unique,
        fill_value=0,           # important when join="outer" and X is sparse
        merge="same",           # keep obs/var columns only if identical across inputs
    )
    return adata_merged

class Plugin(ModelPlugin):
    key = "perturbench"

    def __init__(self, experiment_root=None):
        default = "/gpfs/home/asun/jin_lab/perturbench/src/perturbench/src/perturbench/configs/experiment"
        self.experiment_root = Path(os.environ.get("PERTURBENCH_EXPERIMENT_ROOT", experiment_root or default))

    def prepare_dirs(self, layout):
        ## for now, one config is enough because I can use hydra multirun to run perturbench
        # layout.config_dir("perturbench/linear_additive")
        # layout.config_dir("perturbench/latent_additive")
        #layout.config_dir("perturbench")
        (self.experiment_root / layout.dataset_name).mkdir(parents=True, exist_ok=True)

    def emit_for_split(self,
                       df,
                       split_name,
                       h5ad_path,
                       seed,
                       layout,
                       cfg):

        seed_dir = layout.seed_dir(seed)
        out_dir = layout.config_dir(seed_dir, f"perturbench")
        
        #TODO: fix csv path
        csv_path = out_dir / f"{cfg.data.name}_{split_name}_seed{seed}.csv"
        df[[f"transfer_split_seed{seed}"]].to_csv(csv_path, index=True, index_label="cell_barcode", header=False)

        #HACK: specifics to perturbench
        if isinstance(cfg.data.covariate_key, str):
            covariates = [cfg.data.covariate_key]
        else:
            covariates = cfg.data.covariate_key
        
        # adding batch key as covariate, for example Sample
        if isinstance(cfg.data.batch_key, str):
            covariates.append(cfg.data.batch_key)


        # write yaml files and symlink them
        model_name = "linear_additive"
        yaml_filename = f"{cfg.data.name}_{split_name}_seed{seed}.yaml"
        text = generate_yaml_config(
            "boli_ctx.yaml.j2",
            model_name=model_name, dataset_name=cfg.data.name, split_name=split_name,
            h5ad_path=str(h5ad_path), perturbation_key=cfg.data.perturbation_key,
            covariate_key=covariates, control_value=cfg.data.control_value, csv_path=str(csv_path)
        )
        yaml_path = out_dir / yaml_filename
        yaml_path.write_text(text)

        link_dir = self.experiment_root / cfg.data.name
        link_dir.mkdir(parents=True, exist_ok=True)
        link_path = link_dir / yaml_filename
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(yaml_path, link_path)

        # write prediction dataframe
        test_df = df[df[ f"transfer_split_seed{seed}"] == 'test']
        # Fix: create a list of all columns to select, then use it
        columns_to_select = [cfg.data.perturbation_key] + covariates
        test_combos = test_df[columns_to_select].drop_duplicates()
        prediction_path = out_dir / f"prediction_dataframe.csv"
        test_combos.to_csv(prediction_path, index=False)

        # write sbatch files
        sbatch_filename = f"{cfg.data.name}_{split_name}_seed{seed}.sbatch"
        sbatch = generate_yaml_config(
            "sbatch.j2",
            dataset_name=cfg.data.name,
            yaml_filename=f"{cfg.data.name}_{split_name}_seed{seed}",
            prediction_path=prediction_path,
            out_dir=out_dir
        )
        sbatch_path = out_dir / sbatch_filename
        sbatch_path.write_text(sbatch)

    def visualize_scatterplots(self, cfg):
        model_name = "LatentAdditive"
        
        # STATE latent 2k HVG
        boli = sc.read_h5ad(
            "/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/2025-11-21_12-06-21/data/test_full.h5ad"
        )

        # # Directories, each containing adata_pred.h5ad and adata_real.h5ad
        # adata_dirs = [
        #     f"/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli/seed_{seed}/configs/state/boli_seed{seed}"
        #     for seed in seeds
        # ]

        base_dirs = [
            f"{cfg.output.main_dir}/seed_{seed}/configs/perturbench/latent_additive/"
            for seed in cfg.splitter.seed
        ]

        # ---------------- helpers ----------------
        def parse_cell_pert(cell_pert: str):
            """
            Parse a string like 'L6 CT CTX_Tbr1_0' into:
            cell_type = 'L6 CT CTX'
            perturbation = 'Tbr1'
            """
            parts = cell_pert.split("_")
            cell_type = "_".join(parts[:-2])   # allow spaces encoded as '_'
            perturbation = parts[-2]
            return cell_type, perturbation
        
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

        # ---------------- load and aggregate data from multiple pkl files ----------------
        all_eval_data = []

        for run_dir in base_dirs:
            pkl_res = os.path.join(run_dir, "evaluation/eval.pkl")
            with open(pkl_res, "rb") as f:
                eval_data = pickle.load(f)

            pred_dir = os.path.join(run_dir, "predictions")
            adata_pred = load_and_join_anndatas(pred_dir)
            # TODO: make this a mutable file name / path
            adata_real = sc.read_h5ad("/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli_morph/data/boli_morph_full.h5ad")

            # Create combined cell_type + perturbation index
            cell_type_col = cfg.data.covariate_key  # 'CT'
            condition_col = cfg.data.perturbation_key  # 'Assign'

            # Calculate mean expression per cell_type + perturbation group
            adata_pred.obs['cell_pert'] = (adata_pred.obs[cell_type_col].astype(str) + '_' + 
                                            adata_pred.obs[condition_col].astype(str))
            adata_real.obs['cell_pert'] = (adata_real.obs[cell_type_col].astype(str) + '_' + 
                                            adata_real.obs[condition_col].astype(str))

            # Convert to dense if sparse and create DataFrames
            df_pred = pd.DataFrame(
                adata_pred.X.toarray() if hasattr(adata_pred.X, 'toarray') else adata_pred.X,
                index=adata_pred.obs['cell_pert'],
                columns=genes
            ).groupby(level=0).mean()

            df_ref = pd.DataFrame(
                adata_real.X.toarray() if hasattr(adata_real.X, 'toarray') else adata_real.X,
                index=adata_real.obs['cell_pert'],
                columns=genes
            ).groupby(level=0).mean()

            cell_perts = df_pred.index.tolist()

            # Make a "figures" subdirectory next to the ckpt
            fig_dir = os.path.join(run_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)

            # Plot for each cell_pert individually
            rows = [] 
            for cell_pert in cell_perts:
                if cell_pert not in df_ref.index:
                    continue

                # Plot 1: All genes
                fig, ax = plt.subplots(figsize=(8, 8))
                
                x_vals = df_ref.loc[cell_pert].values
                y_vals = df_pred.loc[cell_pert].values
                
                if np.all(np.isfinite(x_vals)) and np.all(np.isfinite(y_vals)):
                    pearson_r, p_value = pearsonr(x_vals, y_vals)
                else:
                    pearson_r = np.nan
                    p_value = np.nan

                top10_idx = np.argsort(x_vals)[-10:]
                bottom10_idx = np.argsort(x_vals)[:10]
                top_and_bottom_idx = set(top10_idx).union(set(bottom10_idx))

                ax.scatter(x_vals, y_vals, alpha=0.7, color="lightgray")
                ax.plot([x_vals.min(), x_vals.max()], [x_vals.min(), x_vals.max()], 'k--', alpha=0.5)
                ax.set_xlabel("Real Gene Expression")
                ax.set_ylabel(f"Predicted Gene Expression ({model_name})")
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
                safe_cell_pert = str(cell_pert).replace("/", "_")
                fig_path = Path(fig_dir) / "expression" / f"{safe_cell_pert}_all_genes.png"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                log.info("Saving to %s", fig_path)

                plt.tight_layout()
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                
            df_pred = eval_data.aggr["logfc"][model_name].to_df()  # rows: cell_pert, cols: genes
            df_ref  = eval_data.aggr["logfc"]["ref"].to_df()

            pred_dir = os.path.join(run_dir, "predictions")
            adata_pred = load_and_join_anndatas(pred_dir)
            # TODO: make this a mutable file name / path
            adata_real = sc.read_h5ad("/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli_morph/data/boli_morph_full.h5ad")

            # TODO: rename cell type col to covariate col or something
            # NOTE: gene_name_col = None is specific to perturbench using index as names
            df_pred = calculate_logfc_all(adata_pred,
                                          control_value=cfg.data.control_value,
                                          cell_type_col=cfg.data.covariate_key,
                                          condition_col=cfg.data.perturbation_key,
                                          gene_name_col=None)
            
            df_ref = calculate_logfc_all(adata_real,
                                          control_value=cfg.data.control_value,
                                          cell_type_col=cfg.data.covariate_key,
                                          condition_col=cfg.data.perturbation_key)

            cell_perts = df_pred.index.tolist()

            # Make a "figures" subdirectory next to the ckpt
            fig_dir = os.path.join(run_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)

            # Plot for each cell_pert individually
            rows = [] 
            for cell_pert in cell_perts:

                # Plot 1: All genes
                fig, ax = plt.subplots(figsize=(8, 8))
                
                x_vals = df_ref.loc[cell_pert].values
                y_vals = df_pred.loc[cell_pert].values
                
                if np.all(np.isfinite(x_vals)) and np.all(np.isfinite(y_vals)):
                    pearson_r, p_value = pearsonr(x_vals, y_vals)
                else:
                    pearson_r = np.nan
                    p_value = np.nan

                top10_idx = np.argsort(x_vals)[-10:]
                bottom10_idx = np.argsort(x_vals)[:10]
                top_and_bottom_idx = set(top10_idx).union(set(bottom10_idx))

                ax.scatter(x_vals, y_vals, alpha=0.7, color="lightgray")
                ax.plot([x_vals.min(), x_vals.max()], [x_vals.min(), x_vals.max()], 'k--', alpha=0.5)
                ax.set_xlabel("Actual Expression LogFC")
                ax.set_ylabel(f"Predicted Expression LogFC ({model_name})")
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
                safe_cell_pert = str(cell_pert).replace("/", "_")
                fig_path = Path(fig_dir) / "logfc" / f"{safe_cell_pert}_all_genes.png"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                log.info("Saving to %s", fig_path)

                plt.tight_layout()
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                
                cell_type, perturbation = parse_cell_pert(cell_pert)

                rows.append(
                    {
                        "cell_pert": cell_pert,
                        "cell_type": cell_type,
                        "perturbation": perturbation,
                        "pearson_r": pearson_r,
                        "p_value": p_value,
                    }
                )

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
                    safe_cell_pert = str(cell_pert).replace("/", "_")
                    fig_name = f"{safe_cell_pert}_DEGs_only.png"
                    fig_path = os.path.join(deg_dir, fig_name)
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    print(f"Completed plots for {cell_pert}: {len(genes)} total genes, {len(genes_deg)} DEGs")
                else:
                    print(f"No DEGs found for {cell_pert}, skipping DEG plot")

            # Combine rows from all pkl files
            all_eval_data.extend(rows)
        
        # Convert all collected data into a DataFrame
        corr_df = pd.DataFrame(all_eval_data)

        # ---------------- pivot into cell_type x perturbation matrix ----------------
        # If there are multiple entries per (cell_type, perturbation), this will average them
        heatmap_df = corr_df.pivot_table(
            index="cell_type",
            columns="perturbation",
            values="pearson_r",
            aggfunc="mean",
        )

        # Sort axes if desired
        heatmap_df = heatmap_df.sort_index(axis=0)  # sort cell types
        heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)  # sort perts

        # ---------------- plot heatmap ----------------
        plt.figure(
            figsize=(
                max(6, 0.5 * heatmap_df.shape[1]),  # width scales with # perts
                max(6, 0.5 * heatmap_df.shape[0]),  # height scales with # cell types
            )
        )
        sns.heatmap(
            heatmap_df,
            vmin=0,
            vmax=1,
            cmap="viridis",
            annot=False,
            cbar_kws={"label": "Pearson r (pred vs ref logFC)"},
        )
        plt.xlabel("Perturbation")
        plt.ylabel("Cell type")
        plt.title(f"{model_name}: Pearson correlation of logFC (all genes)")
        plt.tight_layout()
        plt.show()
        #plt.savefig("pearson_heatmap_all_genes.pdf", bbox_inches="tight")
        plt.close()
        
