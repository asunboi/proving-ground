# storm/plugins/state/plugin.py
from pathlib import Path
import pandas as pd
from ..base import ModelPlugin
from importlib import resources
from jinja2 import Environment, FileSystemLoader
import logging

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
        sbatch = _generate_from_template(
            "state_sbatch.j2",
            toml_config_path=toml_path,
            perturbation_key=perturbation_key,
            covariate_key=covariate_key,
            control_value=control_value,
            dataset_name=dataset_name,
            output_dir=out_dir,
        )
        sbatch_path = out_dir / sbatch_filename
        sbatch_path.write_text(sbatch)