
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

