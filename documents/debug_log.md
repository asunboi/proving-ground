## Bug ID: 2025-11-18-state-sif-toml-path

**Context**
- Project: storm
- Commit: f9a2393
- Env (conda/module/container): perturbench
- Command: 

**Symptom**
- Error message:
Issue with trying to run `/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/sbatch/state.sbatch`
```
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/toml/test_qual_high_amt_high.toml'
```
even though the file does exist. 
![Alt text](image.png)
- Observed behavior:

**Expected Behavior**
- What should have happened:

**Fixed**
Locally I could run this, so not sure why it wasn't working in sbatch. Was able to see the files in the nodes as well, verified through
```
#!/bin/bash
#SBATCH --time=72:00:00     # walltime
#SBATCH --cpus-per-task=16  # number of cores
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --partition=alphafold,gpu
#SBATCH --gpus=1

module load state

echo "Running on node: $(hostname)"
echo "CWD is: $(pwd)"

python - << 'PY'
import os

p = "/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/toml/test_qual_high_amt_high.toml"
print("toml_path repr:", repr(p))
print("exists?", os.path.exists(p))
print("isfile?", os.path.isfile(p))

if os.path.exists(p):
    with open(p) as f:
        print("First line:", f.readline().strip())
PY
```
Probably an issue with the SIF itself, so checked to see if the image could see it.
```
#!/bin/bash
apptainer exec /tmp/applications/state/0.9.14/bin/state.sif bash -lc '
  echo "Inside container:"
  pwd
  echo "Root:"
  ls /
  echo "Check /gpfs:"
  ls /gpfs || echo "/gpfs does not exist"
  echo "Check your TOML:"
  ls -l /gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/toml/test_qual_high_amt_high.toml || echo "not visible in container"
'
```
which gave the resulting
```
Inside container:
/gpfs/home/asun
Root:
bin   environment  home   lib64   mnt   root  singularity  tmp
boot  etc          lib    libx32  opt   run   srv          usr
dev   gpfs         lib32  media   proc  sbin  sys          var
Check /gpfs:
home
Check your TOML:
ls: cannot access '/gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/toml/test_qual_high_amt_high.toml': No such file or directory
not visible in container
```
it's likely that the container doesn't bind /gpfs/group/jin which is what my /gpfs/home/asun/jin_lab/ redirects to.
Fixed by:
```
apptainer run --nv \
  --bind /gpfs/home/asun:/gpfs/home/asun \
  --bind /gpfs/group/jin:/gpfs/group/jin \
  /tmp/applications/state/0.9.14/bin/state.sif "$@"
```

**Minimal Repro**
```bash
# commands or small script

