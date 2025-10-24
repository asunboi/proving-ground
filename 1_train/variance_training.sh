#!/bin/bash
#SBATCH -J variance_training_5   # Job name
#SBATCH --time=1:00:00     # walltime
#SBATCH --cpus-per-task=4  # number of cores
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_1=glob(*)'
#HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_2=glob(*)'
#HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_3=glob(*)'
#HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_4=glob(*)'
#HYDRA_FULL_ERROR=1 train -m '+experiment/replogle_251008_5=glob(*)'

#HYDRA_FULL_ERROR=1 train -m '+experiment/replogle_251008_1=glob(*)'
#HYDRA_FULL_ERROR=1 train -m '+experiment/replogle_251008_2=glob(*)'
#HYDRA_FULL_ERROR=1 train -m '+experiment/replogle_251008_3=glob(*)'
#HYDRA_FULL_ERROR=1 train -m '+experiment/replogle_251008_4=glob(*)'
HYDRA_FULL_ERROR=1 train -m '+experiment/replogle_251008_5=glob(*)'

