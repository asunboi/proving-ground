# submit.py
import os, subprocess
from omegaconf import OmegaConf
from bin.render_state import sbatch_exports

cfg = OmegaConf.merge(
    OmegaConf.load("conf/splitter.yaml"),
    OmegaConf.load("conf/adapters/state.yaml")
)

env = os.environ.copy()
env.update({
    "TOML_PATH": cfg.state.data.kwargs.toml_config_path,
    "PERT_COL": cfg.state.data.kwargs.pert_col,
    "CELL_TYPE_KEY": cfg.state.data.kwargs.cell_type_key,
    "CONTROL_PERT": cfg.state.data.kwargs.control_pert,
    "OUTDIR": cfg.state.run.outdir,
    "RUN_NAME": cfg.state.run.name,
})

subprocess.run([
    "sbatch", "-J", cfg.state.run.name, "-t", "72:00:00", "-c", "16",
    "--mem", "128G", "-p", "alphafold,gpu", "--gpus", "1",
    "--export", "ALL," + ",".join(f"{k}={v}" for k,v in env.items() if k in
        ["TOML_PATH","PERT_COL","CELL_TYPE_KEY","CONTROL_PERT","OUTDIR","RUN_NAME"]),
    "state_train.sbatch"
], check=True)