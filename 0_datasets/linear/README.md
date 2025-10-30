Basically the same as the linear_hvg directory, but removed `boli = boli[:, boli.var['highly_variable']].copy()` from the linear_processing.py script.

Therefore, running with 32k genes -> predicting 32k genes.

To generate linear_baseline.h5ad, run python linear_processing.py.

