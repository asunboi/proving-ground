# storm/plugins/perturbench/plugin.py
from pathlib import Path
import os
from importlib import resources
from jinja2 import Environment, FileSystemLoader
from ..base import ModelPlugin

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
        layout.config_dir("perturbench/linear_additive")
        layout.config_dir("perturbench/latent_additive")
        (self.experiment_root / layout.dataset_name).mkdir(parents=True, exist_ok=True)

    def emit_for_split(self, df, dataset_name, split_name, h5ad_path, csv_path,
                       perturbation_key, covariate_key, control_value, layout):
        yaml_filename = f"{dataset_name}_{split_name}.yaml"
        for model_name in ("linear_additive", "latent_additive"):
            text = generate_yaml_config(
                "boli_ctx.yaml.j2",
                model_name=model_name, dataset_name=dataset_name, split_name=split_name,
                h5ad_path=str(h5ad_path), perturbation_key=perturbation_key,
                covariate_key=covariate_key, control_value=control_value, csv_path=str(csv_path)
            )
            out_dir = layout.config_dir(f"perturbench/{model_name}")
            yaml_path = out_dir / yaml_filename
            yaml_path.write_text(text)

            link_dir = self.experiment_root / dataset_name
            link_dir.mkdir(parents=True, exist_ok=True)
            link_path = link_dir / yaml_filename
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(yaml_path, link_path)