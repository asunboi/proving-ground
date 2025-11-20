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


call
```
state tx predict --output_dir "test_boli_NTsubset/test_boli_NTsubset_run" --checkpoint "final.ckpt"
```
```
Traceback (most recent call last):
  File "/root/.local/bin/state", line 10, in <module>
    sys.exit(main())
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/state/__main__.py", line 124, in main
    run_tx_predict(args)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/state/_cli/_tx/_predict.py", line 347, in run_tx_predict
    adata_pred.write_h5ad(adata_pred_path)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_core/anndata.py", line 1871, in write_h5ad
    write_h5ad(
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/h5ad.py", line 106, in write_h5ad
    write_elem(f, "obs", adata.obs, dataset_kwargs=dataset_kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/specs/registry.py", line 487, in write_elem
    Writer(_REGISTRY).write_elem(store, k, elem, dataset_kwargs=dataset_kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/utils.py", line 252, in func_wrapper
    return func(*args, **kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/specs/registry.py", line 354, in write_elem
    return write_func(store, k, elem, dataset_kwargs=dataset_kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/specs/registry.py", line 71, in wrapper
    result = func(g, k, *args, **kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/specs/methods.py", line 916, in write_dataframe
    _writer.write_elem(
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/utils.py", line 252, in func_wrapper
    return func(*args, **kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/specs/registry.py", line 354, in write_elem
    return write_func(store, k, elem, dataset_kwargs=dataset_kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/specs/registry.py", line 71, in wrapper
    result = func(g, k, *args, **kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/utils.py", line 312, in func_wrapper
    func(f, k, elem, _writer=_writer, dataset_kwargs=dataset_kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/anndata/_io/specs/methods.py", line 534, in write_vlen_string_array
    f.create_dataset(k, data=elem.astype(str_dtype), dtype=str_dtype, **dataset_kwargs)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/h5py/_hl/group.py", line 186, in create_dataset
    dsid = dataset.make_new_dset(group, shape, dtype, data, name, **kwds)
  File "/root/.local/share/uv/tools/arc-state/lib/python3.10/site-packages/h5py/_hl/dataset.py", line 178, in make_new_dset
    dset_id.write(h5s.ALL, h5s.ALL, data)
  File "h5py/_objects.pyx", line 56, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 57, in h5py._objects.with_phil.wrapper
  File "h5py/h5d.pyx", line 307, in h5py.h5d.DatasetID.write
  File "h5py/_proxy.pyx", line 155, in h5py._proxy.dset_rw
  File "h5py/_conv.pyx", line 442, in h5py._conv.str2vlen
  File "h5py/_conv.pyx", line 96, in h5py._conv.generic_converter
  File "h5py/_conv.pyx", line 247, in h5py._conv.conv_str2vlen
TypeError: Can't implicitly convert non-string objects to strings
Error raised while writing key 'ctrl_cell_barcode' of <class 'h5py._hl.group.Group'> to /obs
```
What I think is happening is that some of the cell types in the splits we choose do not have enough control cells or that state's control cell splitter is not properly handling cases where we have low cell counts and evenly splitting them between val and test. 
I can't be sure of this but I assume this is the case, alternatively we would have to look at the batches / source code logging to make sure this is happening.
We have a couple of cell types with low counts, this is also an issue with perturbench.
# Disscussion 
What should we do with these cell types? Should we only use them as training? Should we completely remove them? Is it important to validate state's performance on celltypes like these?
Currently I am planning on implementing a quick filter to identify cell types with low training counts and assign them as training only. 