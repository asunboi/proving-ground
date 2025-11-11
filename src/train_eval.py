import math, random, numpy as np, pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import wandb

# -----------------------------
# Data containers
# -----------------------------
@dataclass
class Inputs:
    celltype: np.ndarray        # (N,) int64
    perturb: np.ndarray         # (N,) int64
    dataset_id: np.ndarray      # (N,) int64  (0=in-vitro, 1=in-vivo), or more
    covariates: Optional[np.ndarray]  # (N, C) float32 or None
    y: np.ndarray               # (N, G) float32 (targets per gene)

class CellPertDataset(Dataset):
    def __init__(self, inp: Inputs, idx: np.ndarray):
        self.ct = torch.from_numpy(inp.celltype[idx].astype(np.int64))
        self.pt = torch.from_numpy(inp.perturb[idx].astype(np.int64))
        self.ds = torch.from_numpy(inp.dataset_id[idx].astype(np.int64))
        self.cov = None if inp.covariates is None else torch.from_numpy(inp.covariates[idx].astype(np.float32))
        self.y = torch.from_numpy(inp.y[idx].astype(np.float32))

    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i):
        return self.ct[i], self.pt[i], self.ds[i], (self.cov[i] if self.cov is not None else torch.tensor([])), self.y[i]

# -----------------------------
# Models
# -----------------------------
class MLPModel(nn.Module):
    def __init__(self, n_celltypes, n_perts, n_datasets,
                 cov_dim, hidden=512, emb_dim=64, out_dim=1000, p_drop=0.2):
        super().__init__()
        self.ct_emb = nn.Embedding(n_celltypes, emb_dim)
        self.pt_emb = nn.Embedding(n_perts,     emb_dim)
        self.ds_emb = nn.Embedding(n_datasets,  emb_dim)
        in_dim = emb_dim*3 + (cov_dim or 0)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden//2, out_dim)
        )

        # Kaiming init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, ct, pt, ds, cov):
        z = torch.cat([self.ct_emb(ct), self.pt_emb(pt), self.ds_emb(ds)], dim=-1)
        if cov.numel() > 0: z = torch.cat([z, cov], dim=-1)
        return self.net(z)

class TinyTransformer(nn.Module):
    """
    Tokens: [CLS], celltype, perturb, dataset, (optional) pooled covariates.
    We pass through 1 TransformerEncoderLayer, then use CLS for regression head.
    """
    def __init__(self, n_celltypes, n_perts, n_datasets,
                 cov_dim, d_model=128, nhead=4, dim_ff=256, out_dim=1000, p_drop=0.1):
        super().__init__()
        self.ct_emb = nn.Embedding(n_celltypes, d_model)
        self.pt_emb = nn.Embedding(n_perts,     d_model)
        self.ds_emb = nn.Embedding(n_datasets,  d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Project continuous covariates to token if provided
        self.has_cov = cov_dim and cov_dim > 0
        if self.has_cov:
            self.cov_proj = nn.Sequential(
                nn.Linear(cov_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(inplace=True)
            )

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_ff,
                                               dropout=p_drop, activation='gelu',
                                               batch_first=False)  # (S, B, E)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

        # positional enc (tiny fixed)
        self.pos = nn.Parameter(torch.zeros(1, 4 + (1 if self.has_cov else 0), d_model))

    def forward(self, ct, pt, ds, cov):
        B = ct.shape[0]
        tokens = [
            self.cls.expand(-1, B, -1),                       # (1,B,E)
            self.ct_emb(ct).unsqueeze(0),                     # (1,B,E)
            self.pt_emb(pt).unsqueeze(0),                     # (1,B,E)
            self.ds_emb(ds).unsqueeze(0)                      # (1,B,E)
        ]
        if self.has_cov:
            cov_tok = self.cov_proj(cov).unsqueeze(0)         # (1,B,E)
            tokens.append(cov_tok)

        x = torch.cat(tokens, dim=0) + self.pos[:, :len(tokens), :]
        h = self.encoder(x)                                   # (S,B,E)
        cls = h[0]                                            # (B,E)
        y = self.out(cls)                                     # (B,G)
        return y

# -----------------------------
# Autoencoder: encoder/decoder
# -----------------------------
class AEEncoder(nn.Module):
    def __init__(self, in_dim, hidden=1024, latent=64, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden//2, latent)
        )
    def forward(self, x):  # x: (B, G)
        return self.net(x)

class AEDecoder(nn.Module):
    def __init__(self, out_dim, hidden=1024, latent=64, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden//2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden//2, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, z):  # z: (B, latent)
        return self.net(z)

class Autoencoder(nn.Module):
    def __init__(self, in_out_dim, hidden=1024, latent=64, p_drop=0.1):
        super().__init__()
        self.enc = AEEncoder(in_out_dim, hidden, latent, p_drop)
        self.dec = AEDecoder(in_out_dim, hidden, latent, p_drop)
    def forward(self, x):              # returns recon and latent
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z

# ---------------------------------------------
# Supervised head: encoder latent + embeddings
# ---------------------------------------------
class AEHeadModel(nn.Module):
    """
    y -> encoder -> z ; concat with [ct, pt, ds, cov] embeddings -> MLP -> y_pred
    If freeze_encoder=True, encoder weights are not updated during supervised training.
    """
    def __init__(self, encoder: AEEncoder,
                 n_celltypes, n_perts, n_datasets, cov_dim,
                 emb_dim=64, hidden=512, out_dim=1000, freeze_encoder=False, p_drop=0.2):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.ct_emb = nn.Embedding(n_celltypes, emb_dim)
        self.pt_emb = nn.Embedding(n_perts,     emb_dim)
        self.ds_emb = nn.Embedding(n_datasets,  emb_dim)
        in_dim = encoder.net[-1].out_features + emb_dim*3 + (cov_dim or 0)

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden//2, out_dim)
        )

    def forward(self, ct, pt, ds, cov, y):
        with torch.set_grad_enabled(not self.freeze_encoder):
            z = self.encoder(y)  # encode the observed profile
        z = z.detach() if self.freeze_encoder else z
        feats = [z, self.ct_emb(ct), self.pt_emb(pt), self.ds_emb(ds)]
        if cov.numel() > 0: feats.append(cov)
        h = torch.cat(feats, dim=-1)
        return self.head(h)
    
# -----------------------------
# Metrics
# -----------------------------
def pearson_per_gene(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, np.ndarray]:
    # y_*: (N, G)
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    num = (yt * yp).sum(dim=0)
    den = torch.sqrt((yt**2).sum(dim=0) * (yp**2).sum(dim=0) + 1e-12)
    r = (num / den).cpu().numpy()
    r[np.isnan(r)] = 0.0
    return float(np.nanmean(r)), r

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()

# -----------------------------
# Training / eval
# -----------------------------
def train_one_epoch(model, opt, loader, device, l2=0.0):
    model.train()
    total = 0.0
    for ct, pt, ds, cov, y in loader:
        ct, pt, ds, y = ct.to(device), pt.to(device), ds.to(device), y.to(device)
        cov = cov.to(device) if cov.numel() > 0 else torch.empty((y.size(0),0), device=device)
        opt.zero_grad()
        pred = model(ct, pt, ds, cov)
        loss = F.mse_loss(pred, y)
        if l2 > 0:
            l2reg = sum((p**2).sum() for p in model.parameters())
            loss = loss + l2 * l2reg
        loss.backward()
        opt.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for ct, pt, ds, cov, y in loader:
        ct, pt, ds, y = ct.to(device), pt.to(device), ds.to(device), y.to(device)
        cov = cov.to(device) if cov.numel() > 0 else torch.empty((y.size(0),0), device=device)
        pred = model(ct, pt, ds, cov)
        ys.append(y); ps.append(pred)
    y = torch.cat(ys, dim=0); p = torch.cat(ps, dim=0)
    mean_r, r_vec = pearson_per_gene(y, p)
    return {"rmse": rmse(y, p), "pearson_mean": mean_r, "pearson_per_gene": r_vec}

def train_ae_epoch(ae, opt, loader, device, l2=0.0):
    ae.train()
    total = 0.0
    for ct, pt, ds, cov, y in loader:
        y = y.to(device)
        opt.zero_grad()
        xhat, _ = ae(y)
        loss = F.mse_loss(xhat, y)
        if l2 > 0:
            l2reg = sum((p**2).sum() for p in ae.parameters())
            loss = loss + l2 * l2reg
        loss.backward()
        opt.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_ae(ae, loader, device):
    ae.eval()
    ys, yh = [], []
    for _, _, _, _, y in loader:
        y = y.to(device)
        xhat, _ = ae(y)
        ys.append(y); yh.append(xhat)
    y = torch.cat(ys, dim=0); xh = torch.cat(yh, dim=0)
    return {"rmse": torch.sqrt(F.mse_loss(xh, y)).item()}
# -----------------------------
# LOCP split utility
# -----------------------------
def make_locp_split(celltype, perturb, holdout_ct, holdout_pt) -> Tuple[np.ndarray, np.ndarray]:
    mask_hold = (celltype == holdout_ct) & (perturb == holdout_pt)
    idx_test = np.where(mask_hold)[0]
    idx_train = np.where(~mask_hold)[0]
    return idx_train, idx_test

# -----------------------------
# Example driver (replace with your loaders)
# -----------------------------
def main(model_type="mlp",
         hidden=512, emb_dim=64, d_model=128, nhead=4, dim_ff=256,
         batch_size=512, epochs=20, lr=1e-3, weight_decay=1e-5,
         holdout_ct=0, holdout_pt=0,
         ae_hidden=1024, ae_latent=64, ae_epochs=20, freeze_encoder=True):

    wandb.init(
        project="scRNA_LOCP_eval",
        config={
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "holdout_ct": holdout_ct,
            "holdout_pt": holdout_pt,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- Replace this block with your real data loading -------------
    # For demo, we generate synthetic shapes:
    N, COV, G = 20000, 4, 1000               # samples, covariates, genes
    n_celltypes, n_perts, n_datasets = 12, 24, 2

    rng = np.random.default_rng(0)
    celltype = rng.integers(0, n_celltypes, size=N)
    perturb  = rng.integers(0, n_perts,     size=N)
    dataset  = rng.integers(0, n_datasets,  size=N)
    covariates = rng.normal(size=(N, COV)).astype(np.float32)

    # Simulated target (put your real y here; e.g., log1p TPM or Δexpression)
    base = rng.normal(size=(N, G)).astype(np.float32)
    y = base + (celltype[:,None] % 3) * 0.1 + (perturb[:,None] % 5) * 0.05 + dataset[:,None]*0.03
    y = y.astype(np.float32)

    # Standardize covariates (optional but helps)
    cov_scaler = StandardScaler().fit(covariates)
    covariates = cov_scaler.transform(covariates).astype(np.float32)

    inp = Inputs(celltype, perturb, dataset, covariates, y)
    # ---------------------------------------------------------------------

    
    # LOCP split
    idx_train, idx_test = make_locp_split(inp.celltype, inp.perturb, holdout_ct, holdout_pt)
    tr_ds = CellPertDataset(inp, idx_train)
    te_ds = CellPertDataset(inp, idx_test)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    te = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # Build model
    out_dim = inp.y.shape[1]

    if model_type.lower() == "ae":
        ae = Autoencoder(in_out_dim=out_dim, hidden=ae_hidden, latent=ae_latent, p_drop=0.1).to(device)
        opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
        for ep in range(1, ae_epochs+1):
            tr_loss = train_ae_epoch(ae, opt, tr, device)
            if ep % 5 == 0 or ep == 1 or ep == ae_epochs:
                te_metrics = eval_ae(ae, te, device)
                print(f"[AE] epoch {ep:02d} | train_recon_mse {tr_loss:.4f} | test_recon_rmse {te_metrics['rmse']:.4f}")
        # Done: AE is evaluated by reconstruction RMSE
        return

    if model_type.lower() == "aehead":
        # 1) Pretrain AE on training set only
        ae = Autoencoder(in_out_dim=out_dim, hidden=ae_hidden, latent=ae_latent, p_drop=0.1).to(device)
        opt_ae = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
        for ep in range(1, ae_epochs+1):
            tr_loss = train_ae_epoch(ae, opt_ae, tr, device)
            if ep % 5 == 0 or ep == 1 or ep == ae_epochs:
                te_metrics = eval_ae(ae, te, device)
                print(f"[AE] epoch {ep:02d} | train_recon_mse {tr_loss:.4f} | test_recon_rmse {te_metrics['rmse']:.4f}")

        # 2) Supervised head using encoder (optionally frozen)
        enc = ae.enc
        model = AEHeadModel(
            encoder=enc,
            n_celltypes=n_celltypes, n_perts=n_perts, n_datasets=n_datasets,
            cov_dim=(inp.covariates.shape[1] if inp.covariates is not None else 0),
            emb_dim=64, hidden=512, out_dim=out_dim, freeze_encoder=freeze_encoder, p_drop=0.2
        ).to(device)

        # If freezing, only optimize the head + embeddings
        params = (p for n,p in model.named_parameters() if (not freeze_encoder) or ("encoder." not in n))
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)

        for ep in range(1, epochs+1):
            # reuse existing train_one_epoch/evaluate but pass y to forward
            model.train()
            total = 0.0
            for ct, pt, ds, cov, y in tr:
                ct, pt, ds, y = ct.to(device), pt.to(device), ds.to(device), y.to(device)
                cov = cov.to(device) if cov.numel() > 0 else torch.empty((y.size(0),0), device=device)
                opt.zero_grad()
                pred = model(ct, pt, ds, cov, y)
                loss = F.mse_loss(pred, y)
                loss.backward()
                opt.step()
                total += loss.item() * y.size(0)
            tr_mse = total / len(tr.dataset)

            # eval
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for ct, pt, ds, cov, y in te:
                    ct, pt, ds, y = ct.to(device), pt.to(device), ds.to(device), y.to(device)
                    cov = cov.to(device) if cov.numel() > 0 else torch.empty((y.size(0),0), device=device)
                    pred = model(ct, pt, ds, cov, y)
                    ys.append(y); ps.append(pred)
            ycat = torch.cat(ys,0); pcat = torch.cat(ps,0)
            mean_r, _ = pearson_per_gene(ycat, pcat)
            print(f"[AE+Head] epoch {ep:02d} | train_mse {tr_mse:.4f} | test_rmse {rmse(ycat,pcat):.4f} | test_pearson {mean_r:.4f}")
        return


    if model_type.lower() == "mlp":
        model = MLPModel(n_celltypes, n_perts, n_datasets,
                         cov_dim=(inp.covariates.shape[1] if inp.covariates is not None else 0),
                         hidden=hidden, emb_dim=emb_dim, out_dim=out_dim)
    elif model_type.lower() == "attn":
        model = TinyTransformer(n_celltypes, n_perts, n_datasets,
                                cov_dim=(inp.covariates.shape[1] if inp.covariates is not None else 0),
                                d_model=d_model, nhead=nhead, dim_ff=dim_ff, out_dim=out_dim)
    else:
        raise ValueError("model_type must be 'mlp' or 'attn'")

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    for ep in range(1, epochs+1):
        tr_loss = train_one_epoch(model, opt, tr, device)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            te_metrics = evaluate(model, te, device)
            log_dict = {
                "epoch": ep,
                "train_mse": tr_loss,
                "test_rmse": te_metrics['rmse'],
                "test_pearson": te_metrics['pearson_mean'],
            }
            wandb.log(log_dict)
            print(log_dict)

    # Final eval
    te_metrics = evaluate(model, te, device)
    print(f"FINAL [{model_type}]  RMSE={te_metrics['rmse']:.4f}  Pearson(mean)={te_metrics['pearson_mean']:.4f}")
    wandb.finish()

if __name__ == "__main__":
    # Example runs (swap holdout_ct / holdout_pt to cycle LOCP combos)
    # python train_eval.py               # MLP default
    # python train_eval.py attn          # Tiny transformer
    import sys
    mt = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    main(model_type=mt, holdout_ct=2, holdout_pt=7)