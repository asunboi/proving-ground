#!/bin/bash
#SBATCH --time=72:00:00     # walltime
#SBATCH --cpus-per-task=16  # number of cores
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --partition=alphafold,gpu
#SBATCH --gpus=1

module load state

state tx train \
      data.kwargs.toml_config_path="/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/toml/test_qual_high_amt_high.toml" \
      data.kwargs.embed_key="X_hvg" \
      data.kwargs.num_workers=12 \
      data.kwargs.batch_col="Sample" \
      data.kwargs.pert_col="Assign" \
      data.kwargs.cell_type_key="predicted.subclass" \
      data.kwargs.control_pert="NT_0" \
      training.max_steps=40000 \
      training.val_freq=100 \
      training.ckpt_every_n_steps=100 \
      training.batch_size=8 \
      training.lr=1e-4 \
      model.kwargs.cell_set_len=64 \
      model.kwargs.hidden_dim=328 \
      model=pertsets \
      model.kwargs.batch_encoder=True \
      wandb.tags="[test_boli_NTsubset_run]" \
      wandb.entity="asunboi-scripps-research" \
      output_dir="test_boli_NTsubset" \
      name="test_boli_NTsubset_run"

state tx predict --output_dir "test_boli_NTsubset/test_boli_NTsubset_run" --checkpoint "final.ckpt"