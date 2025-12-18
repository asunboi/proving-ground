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
        covariates = [cfg.data.covariate_key, cfg.data.batch_key]

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

    def visualize_scatterplots(seeds):
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
            f"/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli/seed_{i}/configs/perturbench/latent_additive/"
            for seed in seeds
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

            df_pred = eval_data.aggr["logfc"][model_name].to_df()  # rows: cell_pert, cols: genes
            df_ref  = eval_data.aggr["logfc"]["ref"].to_df()

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
                    fig_name = f"{cell_pert}_DEGs_only.png"
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
        
