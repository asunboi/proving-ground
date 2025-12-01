# storm/plugins/perturbench/plugin.py
from pathlib import Path
import os
from importlib import resources
from jinja2 import Environment, FileSystemLoader
from ..base import ModelPlugin
import logging

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

class Plugin(ModelPlugin):
    key = "perturbench"

    def __init__(self, experiment_root=None):
        default = "/gpfs/home/asun/jin_lab/perturbench/src/perturbench/src/perturbench/configs/experiment"
        self.experiment_root = Path(os.environ.get("PERTURBENCH_EXPERIMENT_ROOT", experiment_root or default))

    def prepare_dirs(self, layout):
        ## for now, one config is enough because I can use hydra multirun to run perturbench
        # layout.config_dir("perturbench/linear_additive")
        # layout.config_dir("perturbench/latent_additive")
        layout.config_dir("perturbench")
        (self.experiment_root / layout.dataset_name).mkdir(parents=True, exist_ok=True)

    def emit_for_split(self, df, dataset_name, split_name, h5ad_path, csv_path,
                       perturbation_key, covariate_key, control_value, layout):

        # write yaml files and symlink them
        model_name = "linear_additive"
        yaml_filename = f"{dataset_name}_{split_name}.yaml"
        text = generate_yaml_config(
            "boli_ctx.yaml.j2",
            model_name=model_name, dataset_name=dataset_name, split_name=split_name,
            h5ad_path=str(h5ad_path), perturbation_key=perturbation_key,
            covariate_key=covariate_key, control_value=control_value, csv_path=str(csv_path)
        )
        out_dir = layout.config_dir(f"perturbench")
        yaml_path = out_dir / yaml_filename
        yaml_path.write_text(text)

        link_dir = self.experiment_root / dataset_name
        link_dir.mkdir(parents=True, exist_ok=True)
        link_path = link_dir / yaml_filename
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(yaml_path, link_path)

        # write sbatch files
        sbatch_filename = f"{dataset_name}_{split_name}.sbatch"
        sbatch = generate_yaml_config(
            "sbatch.j2",
            dataset_name=dataset_name,
            yaml_filename=f"{dataset_name}_{split_name}",
        )
        sbatch_path = out_dir / sbatch_filename
        sbatch_path.write_text(sbatch)