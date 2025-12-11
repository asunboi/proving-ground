# storm
- [X] filter to identify cell types with low training counts and assign them as training only

- [X] (aes) make it so that all console outputs are redirected to log, both print statements and errors. 
this is actually not that aesthetic, and more of a mandatory thing. 
resolution detailed in documents/issues/logging.md
resolved in commit 5e75ddf

- [X] generate scripts after storm output
  - [X] perturbench
  - [X] state

- [ ] automatically generate evaluation metrics

- [X] seeded folders for different splitter seeds
can even do this with manual and stuff
configs/
  perturbench/
    seed1/
    seed2/
    seed1_manual/

- [ ] continue refactoring storm into smaller programs / subruns
run storm init to create initial directory with stuff such as data. 
each seeded split will have its own figures, configs, etc. this is all mostly dependent on the splitter seed. 

- [ ] turn storm into a submodule / have its own github repo after completion

## storm: aesthetic
- [ ] (aes) when generating split figures, make it so that whatever is set as control key is on the bottom

## storm: refactor
- [X] refactor splitter and storm into individual packages

- [X] (aes) use jinja2 to render yaml and other templates
finished in Commit 3781434

- [X] refactor storm's save_datasets so that it's mutable in the future and as a seperate function / dataclass
done in commit f9a2393, created save.py to encapuslate plotting and generation of downstream directories

- [X] (aes) rework how the splitter script outputs files / hydra configuration
this was way too vague, but i basically made it so that it outputs something like this:
```
.
└── test
    └── storm
        ├── 2025-11-17_14-44-33
        │   ├── data
        │   ├── figures
        │   ├── latent
        │   ├── linear
        │   ├── splits
        │   └── toml
```
following the format 
```
# output directory, generated dynamically on each run
run:
  dir: outputs/${dataset.name}/${task_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}
```
done in commit 789de8cc5453a6fa49932490f81c59bcda0d4e2a

- [X] refactor storm so that dataset downsampling is a seperate function / does not have to run. Potentially even a seperate command?

- [ ] change the storm hydra yaml, moving grouped dependencies into subdir yaml files

- [X] seperate dependencies by model origins
eg. 
state needs a toml, but not perturbench
perturbench needs splits and yamls, but state doesn't.
both need data and figures/split. 
when modularizing (adding or removing models) we can reduce the bulk / remove redundant models.
for example, there's not really a need to make linear & latent configurations; the only change between the two is model = linear_additive or latent_additive. Therefore this could be overwritten in the submission script since perturbench relies on hydra. 

- [X] modularize the way that storm accepts new models, eg. if I wanted to add linear or latent.
what would I need to do for this?
one big overhead variables file for the dataset
translators for different models input configurations
done in commit 1445d23, set model dependent files as a seperate plugins folder

- [ ] storm run scripts based on a config, with a setting to specify latest or a specific directory
potentially depends on dependencies by model origins. 

- [ ] reduce the bulkiness of storm outputs
for example, it's potentially possible to reduce the # of datasets that are required to be made, but this also depends on how we add new models and whether this is applicable to all models. 
for example, perturbench splitter uses trainingsplit{seed} as a new column to assign training splits to a dataset, so it isn't necessary to create a new folder or anndata if you are using the same object and just testing a new seed / combination of perturbations. 
likewise, it might be possible to use a metadata column to subsample the data, eg. low_qual_low_amt will be TRUE for all cells in that specific experiment.
this wouldn't work for state though as i believe it considers all genes in the input set, so we would have to manually subset etc. 

- [ ] create a handler / wrapper script that submits individual sbatch scripts from a hydra yaml configuration. 

## storm: splitter
- [X] if cell type has perturbations in testing and training, make sure that there are also controls within those splits.
could just use the regular splitter and it'll prob fix this.
actually no in /gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/.hydra/config.yaml we see that even without manual controls the splitter still makes this error. We have to just manually remove cell types probably.

- [X] create visualization for the results of predictions across different phenotypes (cell type x perturbation) similar to the DEG dot plot
look at the variance of results across runs, etc.
resolved in notebooks/visualize_storm_output.ipynb

## storm: cloud
- [ ] containerize environment + code
- [ ] set up EC2 gpu instance and test single training run using data copied from HPC
- [ ] mirror processed datasets to S3
- [ ] switch EC2 run to read from S3
- [ ] add aws s3 sync for processing outputs
- [ ] AWS parallelcluster slurm OR nextflow + AWS batch integration
waiting for hpc cluster is way too slow especially as we start ramping up the number of indiviudal experiments we do. there is an urgent need to figure out how we can integrate EC2 into our system 

- [ ] refactor perturbench configurations so that there are standalone configs but then also batch style configs. 
see p&s hydra for more details.






