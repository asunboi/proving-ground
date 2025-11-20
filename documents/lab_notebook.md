
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

