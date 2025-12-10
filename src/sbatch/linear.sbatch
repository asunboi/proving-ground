#!/bin/bash
#SBATCH --time=72:00:00     # walltime
#SBATCH --cpus-per-task=16  # number of cores
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --partition=alphafold,gpu
#SBATCH --gpus=1

train -m experiment=boli_scramble/boli_scramble_full model=linear_additive,latent_additive