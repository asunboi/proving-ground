# bin/render_state.py
import json, shlex
from omegaconf import OmegaConf

def state_cli(cfg):
    s = cfg.state
    args = [
        "state", "tx", "train",
        f"data.kwargs.toml_config_path={s.data.kwargs.toml_config_path}",
        f"data.kwargs.embed_key={s.data.kwargs.embed_key}",
        f"data.kwargs.num_workers={s.data.kwargs.num_workers}",
        f"data.kwargs.batch_col={s.data.kwargs.batch_col}",
        f"data.kwargs.pert_col={s.data.kwargs.pert_col}",
        f"data.kwargs.cell_type_key={s.data.kwargs.cell_type_key}",
        f"data.kwargs.control_pert={s.data.kwargs.control_pert}",
        f"training.max_steps={s.training.max_steps}",
        f"training.val_freq={s.training.val_freq}",
        f"training.ckpt_every_n_steps={s.training.ckpt_every_n_steps}",
        f"training.batch_size={s.training.batch_size}",
        f"training.lr={s.training.lr}",
        f"model.kwargs.cell_set_len={s.model.kwargs.cell_set_len}",
        f"model.kwargs.hidden_dim={s.model.kwargs.hidden_dim}",
        f"model={s.model.name}",
        f"model.kwargs.batch_encoder={str(s.model.kwargs.batch_encoder)}",
        f"wandb.tags={json.dumps(s.wandb.tags)}",
        f"wandb.entity={s.wandb.entity}",
        f"output_dir={s.run.outdir}",
        f"name={s.run.name}",
    ]
    return " ".join(shlex.quote(a) for a in args)

def sbatch_exports(cfg):
    # Use for Option 1 static sbatch + --export
    s = cfg.state
    return {
        "TOML_PATH": s.data.kwargs.toml_config_path,
        "RUN_NAME": s.run.name,
        "OUTDIR": s.run.outdir,
        "WANDB_TAGS": json.dumps(s.wandb.tags),
        "WANDB_ENTITY": s.wandb.entity,
    }