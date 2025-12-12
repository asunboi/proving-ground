# Dec 11

i want to have a __main__.py for my package that takes in a hydra configuration and sends it to the relevant subpackages. 

eg. if i run storm run or storm init or storm visualize, the base configs + any overrides will still be the same. what would this script look like?

I think that init should call projectlayout and create it for the specific confiiguration. 

i want to automatically run prediction after training, but I need to get the path to the model checkpoint, which is generated in ${hydra:runtime.choices.model}/checkpoints/epoch=7-step=2584.ckpt. the epoch and step are always random though, but there is only 1 file in the checkpoints folder. I need to get the filename set it as a variable. 

# Dec 10

I have code that visualizes a heatmap. the majority of the code is conserved between plugins, with the only difference being how df_pred and df_ref are different for different plugins. 

for example, for STATE, 
```
# Seeds to average over
    seeds = list(range(1, 11))

    # Directories, each containing adata_pred.h5ad and adata_real.h5ad
    adata_dirs = [
        f"/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli/seed_{seed}/configs/state/boli_seed{seed}"
        for seed in seeds
    ]

for run_dir in adata_dirs:
        adata_pred_path = os.path.join(run_dir, "eval_final.ckpt/adata_pred.h5ad")
        adata_real_path = os.path.join(run_dir, "eval_final.ckpt/adata_real.h5ad")

        if not (os.path.exists(adata_pred_path) and os.path.exists(adata_real_path)):
            print(f"Skipping {run_dir}: missing adata_pred.h5ad or adata_real.h5ad")
            continue

        print(f"Processing {run_dir}")

        adata_pred = sc.read_h5ad(adata_pred_path)
        adata_real = sc.read_h5ad(adata_real_path)

        logfc_all_real = calculate_logfc_all(adata_real)
        logfc_all_pred = calculate_logfc_all(adata_pred)

        df_pred = logfc_all_pred  # rows: cell_pert, cols: genes
        df_ref  = logfc_all_real
```

whereas for perturbench:
```
pkl_files = []
for i in range(1,11):
    pkl_files.append(f"/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/boli/seed_{i}/configs/perturbench/latent_additive/evaluation/eval.pkl")

for pkl_res in pkl_files:
    with open(pkl_res, "rb") as f:
        eval_data = pickle.load(f)
    
    df_pred = eval_data.aggr["logfc"][model_name].to_df()  # rows: cell_pert, cols: genes
    df_ref  = eval_data.aggr["logfc"]["ref"].to_df()
```

how can I best code this so that the rest of the code is conserved between plugins, but the loading is specific to each plugin? if this is not possible, just let me know without making it more convoluted.

my plugin class theoretically looks like this
```
class ModelPlugin(ABC):
    key: str  # short name, e.g. "state", "perturbench"

    def prepare_dirs(self, layout) -> None:
        pass  # optional

    @abstractmethod
    def emit_for_split(
        self, df: pd.DataFrame, dataset_name: str, split_name: str,
        h5ad_path: Path,
        perturbation_key: str, covariate_key: str, control_value: str, seed: int, layout
    ) -> None:
        ...

    @abstractmethod
    def visualize_heatmap() -> None:
        pass
``` 

# Dec 4

made a __main__.py

read in multiple pkl_res files and add them all to the heatmap. If there is the same cell type and perturbation, instead take the average for that value. 

i want my Y axis to still be cell types, and basically plot a grid of box plots where the Y axis is always 0-1 pearson, 1 celltype x pert for each plot in the grid, but the whole figure should be 1 thing.

created seeds 1 to 10
working on run.py to submit all 10

how to make it so that for the below code the sbatch file is always submitted with the working directory as the one it is located in 

# Dec 3

make 
output:
  main_dir: /gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/${data.name}/${now:%Y-%m-%d}_${now:%H-%M-%S}

be dependent on splitter.seed, so that it becomes

main_dir: /gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/${data.name}/seed_${splitter.seed}

however, splitter is intialized after self. 

defaults:
  - _self_
  - data: boli_scramble
  - hydra: splitter
  - splitter: subclass
  # experiment configs allow for version control of specific hyperparameters
  # e.g. best hyperparameters for given model and datamodule

  - experiment: null

the reason i am doing all this is because I want my directories to be created with seed_{x} in the name, for example 

output:
  main_dir: /gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/${data.name}/seed_${splitter.seed}

run:
  dir: ${output.main_dir}
sweep:
  dir: ${output.main_dir}

this pattern doesn't work with multiple splitter seeds though, which is why I have to multirun. I would much rather be able to pass in 1 seed, or a range, or a list to my splitter and have the hydra log be inside each one depending on which seed it is. 

if cfg.splitter.seed is a list or a range, run the below for each value in that list or range

splitter.split_covariates(
            seed=cfg.splitter.seed,
            print_split=True,
            max_heldout_fraction_per_covariate=cfg.splitter.max_heldout_fraction_per_covariate,
            max_heldout_covariates=cfg.splitter.max_heldout_covariates,
        )
        

# Dec 2

Visualized the shuffle assign predictions against original (non-shuffle) logfc. Had lab meeting and update.
Xin wants me to try and "break" it more, seeing which conditions cause the model to stop working.

I'm experiencing a fat wave of fatigue or something where I don't really wantt to work, could be due to lack of adderall but also just lack of sleep. I think that if I just rest more and play less games, especially during working hours, I can overcome it.

I'm also experiencing some resistance with starting the later phases of the project, but also this is mainly due to not having a concrete next step / list of things to do. I think I should just take it easy and list out all the todo items, even starting work on the smaller ones first just so I can get the ball rolling again.


# Dec 1

Working on shuffle assign.

# Nov 25

working primarily on the shuffled assign
changed raw_data directory into a data directory

data/
  dataset_name/
    raw/
    processed/

also should work on changing the storm configs into more concrete / seperable things, eg reworking data into a seperate subdir.



# Nov 24

joint meeting at 2, so focusing on showing utility of the current pipeline. The first step is to emulate Seoyeon's experiments, which means fixing / setting up the splitter.

left off working on plugins but not urgent compared to the manual splitter.

finished adding apply_toml_manual_split to splitter.py, still have work to do on refactoring the control splits into it though. 

Since that's added though, the control refactoring can come later. The more important things are cloud > plugins > output visualizations > other refactors

# Nov 21

debugging
```
  File "/gpfs/home/asun/miniforge3/envs/perturbench/lib/python3.11/site-packages/sklearn/model_selection/_split.py", line 2499, in _validate_shuffle_split
    raise ValueError(
ValueError: With n_samples=0, test_size=0.5 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.
```
because i'm trying to replace
```
# splitter.split_covariates(
    #     seed=cfg.splitter.seed,
    #     print_split=True,
    #     max_heldout_fraction_per_covariate=cfg.splitter.max_heldout_fraction_per_covariate,
    #     max_heldout_covariates=cfg.splitter.max_heldout_covariates,
    # )

    # FIX: using perturbench's manual splitter with specified set of holdout covariates, testing to see if this works. 
    # BUG: this probably still wouldn't work because the controls get assigned by split_controls. 
    print(covariates_holdout)

    splitter.split_covariates_manual(
        seed=cfg.splitter.seed,
        covariates_holdout=covariates_holdout,
        print_split=True,
        max_heldout_fraction_per_covariate=cfg.splitter.max_heldout_fraction_per_covariate,
    )
```
the issue is in datasplitter.py from perturbench, so thinking about just completely rehauling it wihtout having to think about their splitter.
when I run
```
for covs, df in adata.obs.groupby(["predicted.subclass"]):
```
i'm getting
('Astro',)
and no output for heldout perts.
when i run
```
for covs, df in adata.obs.groupby("predicted.subclass"):
```
it works as intended.
the current implementation is 
```
for covs, df in self.obs_dataframe.groupby(self.covariate_keys[0]):
	covs = frozenset(covs)
```
and there's no output after the for loop, eg. heldout_perturbation_covariates = [].

currently just went back to default behavior with the split_covariates, but will probably adapt the code into storm as a splitter module seperate from perturbench.

https://github.com/ArcInstitute/state/blob/65953fa23e19859d9f34c607ffea257c7c2ba144/src/state/_cli/_tx/_predict.py#L392

working on refactoring save.py into plugin specific items, but this also extends beyond saving.
currently the things that are specific to the models

**State**
in save
- toml directory and files
- runtime script

**perturbench**
in save
- yaml directory and files
- splits directory
in run
- runtime script

left off working on the plugin refactoring, refer to routine memory tips in chatgpt to continue. 


# Nov 20

in order of priority today:
- [X] look at the analysis output and structure it in a way that can be presented tomorrow
- [ ] refactor storm, look at class / module responsibilities and add translating / overhead capability.
- [ ] store environment in SIF image and try to set up cloud
- [ ] aesthetic improvements to storm

working on the first two simultaneously, since loading and visualizing the dataset is taking a while. 

I want to think about modularizing my code. My package, storm, is meant to be a workflow manager that allows me to compare different models. I have a base configuration for the experiment I want, and then I map that configuration to the input of different models and make sure that I can run them all. Right now, in save.py and in storm.py, I always generate the same output directory and subfolders. But for example, my two models are perturbench and state. Perturbench requires a yaml directory and a splits directory, while state requires a toml directory, and these dependencies are separate from each other. There are some shared directories, such as data, that all models will need. I want to think about ways to refactor this code so that I can drop out and add new models easily, only generating the outputs that each model needs. For example, below is my save.py. 
commit 2e27b30 save.py

I want to calculate the logfc between the control group "Assign" = NT_0 and any other assign groups in a subclass specific manner. eg. only compare CT_SUB to CT_SUB between NT_0 and Bcl11b_0.

Created a notebook visualize_storm_output to look at the results for both state and perturbench results. 

# Nov 19

when I add a new feature, what are some things i want to consider?
where does it live? what does it do? is it an optional / configurable feature? what variables does it need to see?

- [ ] filter to identify cell types with low training counts and assign them as training only
**Where does it live?**
lives in storm.py, because the filter will be a simple yes no cfg check. if yes, then it will subset the data in the main function before the splitter takes place, and we can add the cell types back in after. 
**Is it optional?**
yes, will be managed by the hydra configuration use_training_threshold

Currently we do something like this:
```
# DROP ALL CT WITH LESS THAN X CELLS

mask = adata.obs[covariate_keys[0]].map(adata.obs[covariate_keys[0]].value_counts()) >= 50

adata = adata[mask].copy()
```
I think it would be good to replace the logic here.

35 cell types in the original dataset, but many either lack the cell type control splits or even control cells. Filtering by size is a temporary function but not a great one. If there's no control cells in training, STATE uses a 0 vector but perturbench cannot, therefore we should make sure that there are control cells first.

control cells in training ? -> control cells in val and testing ? -> 
this implies that for the manual control splitting I should check to see the distribution first before using the automated splitter. 
currently it would add some bloat but that's ok. 

turn the below function into something that after splitting, checks to see which cell types located in df[covariate_keys[0]] has 0 in training, or 0 in test or val. If 0 in training, print the cell type and remove it from the dataframe. If 0 in testing or validation, print the cell type and then set transfer_split_seed1 to train for that entire cell type. In the end, print all cell types with 0 in training and the number of them. Alo print the number of cell types with 0 in val and testing and the number of them. 

`manual_controls`

removed dropping CT with minimum cell threshold, implemented check_coverage and check_coverage_adata. Technically this is redundant right now, check_coverage pretty much replaces manual_controls, while check_coverage_adata is a bit better than check_coverage, so I should join them into a function sometime. The keep_covariates_with_train_control function is also antiquated since this is implemented in check_coverage anyways.

Currently this only fixes the control cells, but really it should fix the entire cell type to make sure no perturbations are in test / val. Probably better to do this before the splitter, might be able to change the check_coverage_anndata to reflect this. 

Actually check_coverage already fixes this. 

Currently for the actual script submission itself, I go into the storm output directories and run the sbatch scripts like below
`sbatch /gpfs/home/asun/jin_lab/perturbench/src/sbatch/latent.sbatch`. This generates slurm output files in the directories, for state wandb + wandb logs and test_storm_NTsubset directories, and for linear and latent a logs directory. This is kind of convoluted and not what I really want. 

# Nov 18

left off working in /gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/sbatch, trying to resolve the below issue.
```
ValueError("No controls found for covariate condition {'predicted.subclass': 'Sncg'}.")
full_key: data
```

# Nov 17

working in /gpfs/home/asun/jin_lab/perturbench/studies/perturbench/0_datasets/splitter_test, both run_latent.sh and run_state.sh. Testing to see if there is a way to submit everything as an sbatch script. 

submitting / requesting on multiple nodes, wrote a script to check if all nodes are compatible with the options selected. 

```
  File "/gpfs/home/asun/miniforge3/envs/perturbench/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: 'transfer_split_seed1'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

have to change how specific seeds are engrained into the program. it could also be that I just want to use the same dataset with different splits, and it might be easier to just have one adata with constantly updated metadata?

maybe after creating the initial datasets for downsampling it's possible to just use metadata to inform the models of what fraction I want it to use. 

Not seeing a lot of variance in the splitter. 
```
	•	max_heldout_fraction_per_covariate = 0.6
	•	max_heldout_covariates = 3
	•	5 perturbations
	•	21 cell types (covariate combinations)
```
```
With 5 perturbations, that means the absolute maximum number of (perturbation, cell-type) pairs that can ever be held out is:

5 perturbations × 3 covariates/perturbation = 15 held-out pairs

Total possible pairs (if fully crossed) are:

5 perturbations × 21 cell types = 105 pairs

So at a global level, you can only ever mark 15 / 105 ≈ 14% of all pairs as held out.

Spread across 21 cell types, that’s less than 1 held-out perturbation per cell type on average. So it is actually expected that most cell types end up with zero held-out perturbations → all training.

This happens even before you worry about the random sampling details.

If your mental model was “60% of perturbations per cell type should be held out,” then with only 5 perturbations and 21 cell types, you’d need roughly:

0.6 × 5 perts × 21 cell types = 63 held-out pairs

But your max_heldout_covariates = 3 caps you at 15, so the 0.6 is purely an upper bound, not a target that the algorithm could realistically reach.
```

Changed the splitter for subclass to max covariates = 13 or so. 

