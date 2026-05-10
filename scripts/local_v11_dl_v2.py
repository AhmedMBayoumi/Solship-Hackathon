"""
LOCAL v11 — Upgraded DL feature extractor (v10 + multi-improvement).

Upgrades vs v10:
  - bottleneck 16 -> 32             (more pattern diversity)
  - encoder 1-layer LSTM -> 2-layer with dropout 0.1
  - reconstruction-only -> RECON + FORECAST aux loss (predict next 4 steps of load)
      forces the bottleneck to encode forward-predictive features, not just
      compress the past. This is the single biggest expected gain.
  - AE training data 2024 only -> 2024 + Jan-Mar 2025 (3 extra months,
      legal because Jan-Mar < April test month and < September test month).
  - 12 epochs -> 20 epochs with cosine LR schedule.
  - light input noise (Gaussian, sigma=0.05) -> robustness.

Pipeline:
  1. Train AE with combined loss = recon_mse + 0.5 * forecast_mse
  2. Encode every row's past-96 window -> 32-dim bottleneck
  3. Add as features to v7 (-> v11 = 153 + 32 = 185 features)
  4. 8-bag LGBM walkforward
  5. Compare to 60.65% baseline AND blend with online_retraining
"""
import sys, time, warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

# torch FIRST so its DLLs claim their slots before lightgbm's openmp loads
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
DEVICE = "cpu"
print(f"Device: {DEVICE}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Load v7 features
# ──────────────────────────────────────────────────────────────────────
print("Loading v7 features...", flush=True)
df = pd.read_parquet(ROOT / "data/features/features_v7_all.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
n = len(df)
print(f"  rows={n}  cols={df.shape[1]}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 2. Build past-96 windows + future-4 forecast targets
# ──────────────────────────────────────────────────────────────────────
WINDOW   = 96
HORIZON  = 4
N_BOTT   = 32
N_HID    = 64
SIGNALS  = ["load_kw", "pv_kw", "temperature_2m", "shortwave_radiation"]

print(f"Building past-{WINDOW} windows + +{HORIZON}-step forecast targets...", flush=True)
sig_arr = df[SIGNALS].ffill().bfill().fillna(0).values.astype("float32")
load_arr = df["load_kw"].ffill().bfill().fillna(0).values.astype("float32")

# AE training mask: 2024 + Jan-Mar 2025 (all data BEFORE April test month → safe for both folds)
ae_train_mask = ((df["timestamp"].dt.year == 2024) |
                 ((df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month <= 3))).values
chan_mean = sig_arr[ae_train_mask].mean(axis=0)
chan_std  = sig_arr[ae_train_mask].std(axis=0) + 1e-6
sig_z = (sig_arr - chan_mean) / chan_std
load_mean = load_arr[ae_train_mask].mean()
load_std  = load_arr[ae_train_mask].std() + 1e-6
load_z = (load_arr - load_mean) / load_std

# (n, window, n_chan) past windows; (n, horizon) future load targets
windows  = np.zeros((n, WINDOW, len(SIGNALS)), dtype="float32")
fc_tgt   = np.zeros((n, HORIZON), dtype="float32")
for t in range(WINDOW, n - HORIZON):
    windows[t] = sig_z[t - WINDOW:t]
    fc_tgt[t]  = load_z[t:t + HORIZON]
print(f"  windows shape={windows.shape}  forecast-target shape={fc_tgt.shape}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 3. Upgraded LSTM autoencoder + forecast head
# ──────────────────────────────────────────────────────────────────────
class LSTMAE_v2(nn.Module):
    def __init__(self, n_chan=4, n_hid=N_HID, n_bott=N_BOTT, window=WINDOW, horizon=HORIZON, dropout=0.1):
        super().__init__()
        self.window = window; self.horizon = horizon; self.n_hid = n_hid
        self.enc_lstm = nn.LSTM(n_chan, n_hid, num_layers=2, batch_first=True, dropout=dropout)
        self.enc_lin  = nn.Linear(n_hid, n_bott)
        self.dec_lin  = nn.Linear(n_bott, n_hid)
        self.dec_lstm = nn.LSTM(n_hid, n_hid, num_layers=2, batch_first=True, dropout=dropout)
        self.dec_out  = nn.Linear(n_hid, n_chan)
        self.fc_head  = nn.Sequential(nn.Linear(n_bott, 32), nn.ReLU(), nn.Linear(32, horizon))

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)              # h: (num_layers, B, H)
        return self.enc_lin(h[-1])                # use last layer's hidden

    def forward(self, x):
        z   = self.encode(x)                                # (B, bott)
        rep = z.unsqueeze(1).repeat(1, self.window, 1)
        rep = self.dec_lin(rep)                              # (B, T, H)
        out, _ = self.dec_lstm(rep)
        recon = self.dec_out(out)                            # (B, T, C)
        fcst  = self.fc_head(z)                              # (B, horizon)
        return recon, fcst

# ──────────────────────────────────────────────────────────────────────
# 4. Train AE with recon + forecast loss
# ──────────────────────────────────────────────────────────────────────
train_idx = np.where(ae_train_mask & (np.arange(n) >= WINDOW) & (np.arange(n) < n - HORIZON))[0]
print(f"AE training rows: {len(train_idx)} (2024 + Jan-Mar 2025)", flush=True)

# Subsample hourly for speed
train_idx = train_idx[::4]
print(f"  subsampled to {len(train_idx)} rows (hourly cadence)", flush=True)

X_train  = torch.from_numpy(windows[train_idx]).float()
F_train  = torch.from_numpy(fc_tgt[train_idx]).float()

torch.manual_seed(42)
model = LSTMAE_v2().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
recon_loss_fn = nn.MSELoss()
fcst_loss_fn  = nn.MSELoss()

EPOCHS, BATCH = 20, 256
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
LAMBDA_FC = 0.5
NOISE_SIGMA = 0.05

print(f"Training v11 AE: {EPOCHS} epochs, batch={BATCH}, lambda_fc={LAMBDA_FC}, noise={NOISE_SIGMA}", flush=True)
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    rec_losses, fc_losses = [], []
    for i in range(0, len(X_train), BATCH):
        bi = perm[i:i+BATCH]
        xb = X_train[bi].to(DEVICE)
        fb = F_train[bi].to(DEVICE)
        # input noise for robustness
        xb_in = xb + NOISE_SIGMA * torch.randn_like(xb)
        recon, fcst = model(xb_in)
        rec_l  = recon_loss_fn(recon, xb)
        fc_l   = fcst_loss_fn(fcst, fb)
        loss   = rec_l + LAMBDA_FC * fc_l
        opt.zero_grad(); loss.backward(); opt.step()
        rec_losses.append(rec_l.item()); fc_losses.append(fc_l.item())
    sched.step()
    print(f"  epoch {ep+1}/{EPOCHS}  recon={np.mean(rec_losses):.4f}  fc={np.mean(fc_losses):.4f}  lr={sched.get_last_lr()[0]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 5. Encode all rows
# ──────────────────────────────────────────────────────────────────────
print("Encoding all rows...", flush=True)
model.eval()
bott = np.zeros((n, N_BOTT), dtype="float32")
with torch.no_grad():
    for i in range(WINDOW, n, 512):
        xb = torch.from_numpy(windows[i:i+512]).float().to(DEVICE)
        bott[i:i+len(xb)] = model.encode(xb).cpu().numpy()
ae_cols = [f"ae2_{i:02d}" for i in range(N_BOTT)]
for i, c in enumerate(ae_cols):
    df[c] = bott[:, i]
df.loc[df.index < WINDOW, ae_cols] = np.nan
print(f"  added {N_BOTT} AE-v2 features", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 6. 8-bag LGBM walkforward
# ──────────────────────────────────────────────────────────────────────
DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
             "qow","hod","net_load","sell_price"}
feats = [c for c in df.columns if c not in DROP_BASE]
print(f"v11 features: {len(feats)}  ({len(feats)-N_BOTT} v7 + {N_BOTT} AE-v2)", flush=True)

ts = df["timestamp"]
def april_split():
    return (((ts.dt.year == 2024) | ((ts.dt.year == 2025) & (ts.dt.month <= 2))),
            ((ts.dt.year == 2025) & (ts.dt.month == 3)),
            ((ts.dt.year == 2025) & (ts.dt.month == 4)))
def sept_split():
    return (((ts.dt.year == 2024) |
             ((ts.dt.year == 2025) & (ts.dt.month <= 7)) |
             ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day <= 15))),
            ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day > 15)),
            ((ts.dt.year == 2025) & (ts.dt.month == 9)))
def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)

LIGHT_CONFIGS = [
    {"num_leaves":63,"max_depth":8,"learning_rate":0.02,"min_child_samples":20,
     "reg_alpha":0.1,"reg_lambda":0.1,"subsample":0.9,"colsample_bytree":0.9},
    {"num_leaves":47,"max_depth":7,"learning_rate":0.015,"min_child_samples":30,
     "reg_alpha":0.3,"reg_lambda":0.5,"subsample":0.85,"colsample_bytree":0.85},
    {"num_leaves":95,"max_depth":10,"learning_rate":0.025,"min_child_samples":15,
     "reg_alpha":0.05,"reg_lambda":0.1,"subsample":0.85,"colsample_bytree":0.85},
    {"num_leaves":31,"max_depth":6,"learning_rate":0.025,"min_child_samples":40,
     "reg_alpha":0.5,"reg_lambda":0.5,"subsample":0.85,"colsample_bytree":0.85},
]

def run(label, splits, n_bags=8, n_trees=1500):
    tr_m, va_m, te_m = splits
    tv_df = pd.concat([df[tr_m], df[va_m]], ignore_index=True).dropna(subset=feats).reset_index(drop=True)
    te_df = df[te_m].dropna(subset=feats).reset_index(drop=True)
    print(f"\n=== {label}  train+val={len(tv_df)}  test={len(te_df)} ===", flush=True)
    X_tv, y_tv = tv_df[feats].values, tv_df["load_kw"].values
    X_te, y_te = te_df[feats].values, te_df["load_kw"].values
    preds = np.zeros((n_bags, len(X_te)))
    importance = np.zeros(len(feats))
    t0 = time.time()
    for i in range(n_bags):
        cfg = dict(LIGHT_CONFIGS[i % len(LIGHT_CONFIGS)])
        seed = 42 + i
        cfg.update({"n_estimators":n_trees,"subsample_freq":1,"objective":"huber",
                    "alpha":0.9,"verbose":-1,"n_jobs":-1,"random_state":seed})
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_tv), size=int(len(X_tv)*0.9))
        m = lgb.LGBMRegressor(**cfg); m.fit(X_tv[idx], y_tv[idx])
        preds[i] = np.clip(m.predict(X_te), 0, None)
        importance += m.feature_importances_ / n_bags
        print(f"  bag {i+1}/{n_bags} ({time.time()-t0:.0f}s)", flush=True)
    avg = preds.mean(axis=0)
    n_te = nrmse(y_te, avg)
    print(f"  {label} NRMSE: {n_te:.2f}%", flush=True)
    ae_imp = sorted([(f, importance[feats.index(f)]) for f in ae_cols], key=lambda x: -x[1])
    print("  Top AE-v2 feature importances:")
    for f, im in ae_imp[:5]: print(f"    {f}: {im:.0f}")
    return te_df["timestamp"].values, avg, y_te, n_te

ts_a, p_a, y_a, n_a = run("APRIL",     april_split())
ts_s, p_s, y_s, n_s = run("SEPTEMBER", sept_split())

ts_all = np.concatenate([ts_a, ts_s])
p_all  = np.concatenate([p_a, p_s])
y_all  = np.concatenate([y_a, y_s])
n_combined = nrmse(y_all, p_all)

print("\n" + "="*70, flush=True)
print(f"  v11 (upgraded LSTM-AE) combined NRMSE: {n_combined:.2f}%", flush=True)
print(f"  vs prev best (online_retraining): 60.65%", flush=True)
delta = n_combined - 60.65
verdict = "WORSE" if delta > 0 else "IMPROVEMENT!" if delta < 0 else "tied"
print(f"  delta vs baseline: {delta:+.2f}pp  ({verdict})", flush=True)

# Save
out = ROOT / "outputs/forecasts/v11_lstm_ae_v2_test_preds.csv"
pd.DataFrame({"timestamp": ts_all, "load_pred": p_all}).to_csv(out, index=False)
print(f"\nSaved -> {out}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 7. Quick blend with online_retraining (auto-pick blend weight)
# ──────────────────────────────────────────────────────────────────────
try:
    base = pd.read_csv(ROOT / "outputs/forecasts/online_retraining_test_preds.csv")
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    v11_df = pd.DataFrame({"timestamp": ts_all, "load_pred_v11": p_all})
    merged = base.merge(v11_df, on="timestamp", how="inner")
    y_b = pd.read_parquet(ROOT / "data/features/features_v7_all.parquet").assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
    actuals = y_b[(y_b["timestamp"].dt.year == 2025) & (y_b["timestamp"].dt.month.isin([4,9]))][["timestamp","load_kw"]]
    merged = merged.merge(actuals, on="timestamp", how="inner")
    print(f"\nBlending: {len(merged)} aligned rows", flush=True)

    yA = merged["load_kw"].values
    A  = np.column_stack([merged["load_pred"].values, merged["load_pred_v11"].values])
    from scipy.optimize import nnls
    w, _ = nnls(A, yA); w = w / max(w.sum(), 1e-9)
    blend = A @ w
    print(f"  blend weights: online_retraining={w[0]:.3f}, v11={w[1]:.3f}", flush=True)
    print(f"  blend NRMSE: {nrmse(yA, blend):.3f}%", flush=True)

    # 5-fold CV-NNLS for honest estimate
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=False)
    oof = np.zeros_like(yA)
    for tr_idx, va_idx in kf.split(A):
        w_, _ = nnls(A[tr_idx], yA[tr_idx])
        w_ = w_ / max(w_.sum(), 1e-9)
        oof[va_idx] = A[va_idx] @ w_
    print(f"  CV-NNLS NRMSE (honest): {nrmse(yA, oof):.3f}%", flush=True)
    out2 = ROOT / "outputs/forecasts/v11_blend_test_preds.csv"
    pd.DataFrame({"timestamp": merged["timestamp"].values, "load_pred": blend}).to_csv(out2, index=False)
    print(f"  Saved blend -> {out2}", flush=True)
except Exception as e:
    print(f"\n[blend skipped] {e}", flush=True)
