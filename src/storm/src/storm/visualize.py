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
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

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

def visualize_heatmap():
    
    # ---------------- I/O + setup ----------------
    # Seeds to average over
    seeds = list(range(1, 11))

    # Directories, each containing adata_pred.h5ad and adata_real.h5ad
    adata_dirs = [
        f"/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli/seed_{seed}/configs/state/boli_seed{seed}/eval_final.ckpt"
        for seed in seeds
    ]

    adata_pred = sc.read_h5ad("/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/2025-11-19_16-14-33/test_storm_NTsubset/test_storm_NTsubset_run/eval_final.ckpt/adata_pred.h5ad")
    adata_real = sc.read_h5ad("/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/2025-11-19_16-14-33/test_storm_NTsubset/test_storm_NTsubset_run/eval_final.ckpt/adata_real.h5ad")

    logfc_all_real = calculate_logfc_all(adata_real)
    logfc_all_pred = calculate_logfc_all(adata_pred)

    # ---------------- I/O + setup ----------------
    # STATE latent 2k HVG
    boli = sc.read_h5ad(
        "/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/2025-11-21_12-06-21/data/test_full.h5ad"
    )

    model_name="STATE"
    genes = boli.var["gene_name"].values

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

    # (Optional) if you want to keep the DEG loader for later use
    def load_deg_genes(cell_pert: str):
        """
        Load DEG genes for a specific cell type and perturbation.
        Modify the path pattern to match your file naming convention.
        """
        parts = cell_pert.split("_")
        cell_type = "_".join(parts[:-2])   # e.g., "L6 CT CTX"
        perturbation = parts[-2]           # e.g., "Tbr1"
        
        deg_file = (
            f"/gpfs/home/asun/jin_lab/perturbench/raw_data/"
            f"top100_ctx_{perturbation.lower()}_{cell_type.replace(' ', '_')}.csv"
        )
        
        try:
            deg_df = pd.read_csv(deg_file)
            return set(deg_df["names"].tolist())
        except FileNotFoundError:
            print(f"Warning: DEG file not found for {cell_pert}: {deg_file}")
            return set()


    # ---------------- load and aggregate over multiple adata dirs ----------------
    all_eval_data = []

    for run_dir in adata_dirs:
        adata_pred_path = os.path.join(run_dir, "adata_pred.h5ad")
        adata_real_path = os.path.join(run_dir, "adata_real.h5ad")

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

        # ---------------- compute Pearson r per cell_pert (all genes) ----------------
        for cell_pert in cell_perts:
            # actual vs predicted logFC for this cell_pert, across all genes
            x_vals = df_ref.loc[cell_pert].values
            y_vals = df_pred.loc[cell_pert].values

            if np.all(np.isfinite(x_vals)) and np.all(np.isfinite(y_vals)):
                pearson_r, p_value = pearsonr(x_vals, y_vals)
            else:
                pearson_r = np.nan
                p_value = np.nan

            cell_type, perturbation = parse_cell_pert(cell_pert)
            all_eval_data.append(
                {
                    "cell_pert": cell_pert,
                    "cell_type": cell_type,
                    "perturbation": perturbation,
                    "pearson_r": pearson_r,
                    "p_value": p_value,
                }
            )

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
    # plt.savefig("pearson_heatmap_all_genes.pdf", bbox_inches="tight")
    plt.close()

def visualize_scatterplots():
    model_name = "STATE"
    
    # STATE latent 2k HVG
    boli = sc.read_h5ad(
        "/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/2025-11-21_12-06-21/data/test_full.h5ad"
    )

    # ---------------- I/O + setup ----------------
    # Seeds to average over
    seeds = list(range(1, 11))

    # Directories, each containing adata_pred.h5ad and adata_real.h5ad
    adata_dirs = [
        f"/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli/seed_{seed}/configs/state/boli_seed{seed}/eval_final.ckpt"
        for seed in seeds
    ]
    
    genes = boli.var["gene_name"].values

    cell_perts = df_pred.index.tolist()

    def load_deg_genes(cell_pert):
        """
        Load DEG genes for a specific cell type and perturbation.
        Modify the path pattern to match your file naming convention.
        """
        # Parse cell_pert to extract cell type and perturbation
        # Example: "L6 CT CTX_Tbr1_0" -> cell_type="L6_CT_CTX", perturbation="Tbr1"
        parts = cell_pert.split("_")
        cell_type = "_".join(parts[:-2])  # e.g., "L6 CT CTX"
        perturbation = parts[-2]  # e.g., "Tbr1"
        
        # Construct the DEG file path
        # Adjust this pattern to match your actual file naming convention    
        deg_file = f"/gpfs/home/asun/jin_lab/perturbench/raw_data/sig_deg/{cell_type.replace(' ', '_')}_Assign{perturbation}_0.csv"

        try:
            deg_df = pd.read_csv(deg_file)
            filtered = deg_df.loc[deg_df["adj_pval"] < 0.05, "Gene"]
            return set(filtered.dropna().astype(str).tolist())
        except FileNotFoundError:
            print(f"Warning: DEG file not found for {cell_pert}: {deg_file}")
            return set()

    for run_dir in adata_dirs:
        adata_pred_path = os.path.join(run_dir, "adata_pred.h5ad")
        adata_real_path = os.path.join(run_dir, "adata_real.h5ad")

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
            
            plt.tight_layout()
            #plt.savefig(f"{cell_pert}_all_genes.png", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
            # Plot 2: DEGs only for this specific cell_pert
            deg_genes = load_deg_genes(cell_pert)
            
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
                #plt.savefig(f"{cell_pert}_DEGs_only.png", dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                
                print(f"Completed plots for {cell_pert}: {len(genes)} total genes, {len(genes_deg)} DEGs")
            else:
                print(f"No DEGs found for {cell_pert}, skipping DEG plot")


