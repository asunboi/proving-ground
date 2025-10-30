import os
os.chdir("/gpfs/home/asun/jin_lab/perturbench/0_datasets")
print(os.getcwd())

import sys
sys.path.append(os.path.abspath('..'))

import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

boli = sc.read_h5ad('/gpfs/group/jin/skim/STATE/state/test_boli_NTsubset/processed_data/Boli_Perturb_CTX_edit_NTsubset_hvg.h5ad')

boli_hvg = boli[:, boli.var['highly_variable']].copy()

# start with default "train"
boli_hvg.obs["transfer_split_seed1"] = "train"

# condition = TBR1 or SATB2  AND  cell_type = L6_CT_CTX → "test"
mask_test = (
    boli_hvg.obs["Assign"].isin(["Tbr1_0", "Satb2_0"])
    & (boli_hvg.obs["predicted.subclass"] == "L6 CT CTX")
)
boli_hvg.obs.loc[mask_test, "transfer_split_seed1"] = "test"

# condition = XPO7  AND  cell_type = L6_CT_CTX → "val"
mask_val = (
    (boli_hvg.obs["Assign"] == "Xpo7_0")
    & (boli_hvg.obs["predicted.subclass"] == "L6 CT CTX")
)
boli_hvg.obs.loc[mask_val, "transfer_split_seed1"] = "val"

# Create mask for control cells only
control_mask = boli_hvg.obs["Assign"] == "NT_0"

# Train: Batch 1 Sample 1 and 2 (controls only)
train_mask = control_mask & boli_hvg.obs['BioSamp'].str.contains('batch1_samp[12]', case=False, regex=True, na=False)
boli_hvg.obs.loc[train_mask, 'transfer_split_seed1'] = 'train'

# Val: Batch 2 mouse 1 and 2 (controls only)
val_mask = control_mask & boli_hvg.obs['BioSamp'].str.contains('batch2_mouse[12]', case=False, regex=True, na=False)
boli_hvg.obs.loc[val_mask, 'transfer_split_seed1'] = 'val'

# Test: Batch 2 mouse 3 (controls only)
test_mask = control_mask & boli_hvg.obs['BioSamp'].str.contains('batch2_mouse3', case=False, regex=True, na=False)
boli_hvg.obs.loc[test_mask, 'transfer_split_seed1'] = 'test'

# Find cell types that have at least one instance meeting both conditions
valid_cell_types = boli_hvg.obs[
    (boli_hvg.obs["transfer_split_seed1"] == "train") & 
    (boli_hvg.obs["Assign"] == "NT_0")
]["predicted.subclass"].unique()

print(f"Cell types with train + NT_0: {len(valid_cell_types)}")
print(f"Valid cell types: {valid_cell_types}")

# Filter to keep only those cell types
boli_hvg_filtered = boli_hvg[boli_hvg.obs["predicted.subclass"].isin(valid_cell_types)].copy()

print(f"\nOriginal shape: {boli_hvg.shape}")
print(f"Filtered shape: {boli_hvg_filtered.shape}")
print(f"Removed {boli_hvg.n_obs - boli_hvg_filtered.n_obs} cells")

boli_hvg_filtered.write_h5ad("/gpfs/home/asun/jin_lab/perturbench/0_datasets/linear/boli_hvg.h5ad")

# Save CSV with transfer split information
out = boli_hvg_filtered.obs[["transfer_split_seed1"]].copy()
out.to_csv("/gpfs/home/asun/jin_lab/perturbench/0_datasets/linear/linear_baseline_split_ctrls.csv", index=True, index_label="cell_barcode", header=False)