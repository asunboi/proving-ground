# Nov 17

overall
- [ ] refactor splitter and storm into individual packages

src/storm.py
- [ ] (aes) use jinja2 to render yaml and other templates

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

- [ ] (aes) when generating split figures, make it so that whatever is set as control key is on the bottom

- [ ] generate scripts after storm output

- [ ] refactor storm so that dataset downsampling is a seperate function / does not have to run. Potentially even a seperate command?

- [ ] create a handler / wrapper script that submits individual sbatch scripts from a hydra yaml configuration. 