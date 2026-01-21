import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os

def main(cfg: DictConfig):
    seeds = cfg.splitter.seed  # List of seeds
    models = cfg.models  # List of models
    main_dir = Path(cfg.output.main_dir)
    dataset_name = cfg.data.name

    # Function to submit the sbatch script for a given seed and model
    def submit_sbatch(seed: int, model: str):
        sbatch_path = main_dir / f"seed_{seed}" / "configs" / model / f"{dataset_name}_full_seed{seed}.sbatch"

        if sbatch_path.exists():
            # run from the directory that the sbatch file is located in
            sbatch_dir = sbatch_path.parent
            os.chdir(sbatch_dir)

            print(f"Submitting {sbatch_path}...")
            subprocess.run(["sbatch", str(sbatch_path)], check=True)
        else:
            print(f"Error: {sbatch_path} does not exist.")

    # Loop through each seed and model and submit the sbatch job
    for seed in seeds:
        for model in models:
            submit_sbatch(seed, model)

if __name__ == "__main__":
    main()