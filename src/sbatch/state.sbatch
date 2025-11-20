#!/bin/bash
#SBATCH --time=72:00:00     # walltime
#SBATCH --cpus-per-task=16  # number of cores
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --partition=alphafold,gpu
#SBATCH --gpus=1

apptainer run --nv \
  --bind /gpfs/home/asun:/gpfs/home/asun \
  --bind /gpfs/group/jin:/gpfs/group/jin \
  /tmp/applications/state/0.9.14/bin/state.sif \
      tx train \
      data.kwargs.toml_config_path="/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-19_16-14-33/toml/test_qual_high_amt_high.toml" \
      data.kwargs.embed_key="X_hvg" \
      data.kwargs.num_workers=12 \
      data.kwargs.batch_col="Sample" \
      data.kwargs.pert_col="Assign" \
      data.kwargs.cell_type_key="predicted.subclass" \
      data.kwargs.control_pert="NT_0" \
      training.max_steps=5000 \
      training.val_freq=100 \
      training.ckpt_every_n_steps=100 \
      training.batch_size=8 \
      training.lr=1e-4 \
      model.kwargs.cell_set_len=64 \
      model.kwargs.hidden_dim=328 \
      model=pertsets \
      model.kwargs.batch_encoder=True \
      wandb.tags="[test_storm_NTsubset_run]" \
      wandb.entity="asunboi-scripps-research" \
      output_dir="test_storm_NTsubset" \
      name="test_storm_NTsubset_run"

apptainer run --nv \
  --bind /gpfs/home/asun:/gpfs/home/asun \
  --bind /gpfs/group/jin:/gpfs/group/jin \
  /tmp/applications/state/0.9.14/bin/state.sif \
      tx predict \
      --output_dir "test_storm_NTsubset/test_storm_NTsubset_run" \
      --checkpoint "final.ckpt"