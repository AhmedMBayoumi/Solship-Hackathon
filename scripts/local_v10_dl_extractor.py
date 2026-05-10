"""
LOCAL v10 — User's DL feature-extractor idea (TRUE PyTorch LSTM autoencoder).

Architecture:
  STEP 1: LSTM autoencoder trained on 2024 ONLY.
          input  = past 96-step window of [load_kw, pv_kw, temperature_2m, shortwave_radiation]
          encoder LSTM 4 -> 32 -> bottleneck 16
          decoder LSTM 16 -> 32 -> 4
          loss   = MSE reconstruction
  STEP 2: Encode the past-96 window at every timestep -> 16-dim "regime vector".
          NO future leakage (window strictly past).
  STEP 3: Concatenate the 16 LSTM-AE features with v7 tabular features (153 cols).
          Train 8-bag LGBM with our walkforward design:
            April : train 2024+Jan-Mar 2025 -> predict April
            Sept  : train 2024+Jan-Aug 2025 -> predict September
  STEP 4: Combined NRMSE vs 60.65% baseline.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Load v7 features
# ──────────────────────────────────────────────────────────────────────
print("Loading v7 features...", flush=True)
df = pd.read_parquet(ROOT / "data/features/features_v7_all.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"  rows={len(df)}  cols={df.shape[1]}", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 2. Build past-96-step windows of [load, pv, temp, rad]
# ──────────────────────────────────────────────────────────────────────
WINDOW   = 96
N_BOTT   = 16
N_HID    = 32
SIGNALS  = ["load_kw", "pv_kw", "temperature_2m", "shortwave_radiation"]

print(f"Building past-{WINDOW} windows of {SIGNALS}...", flush=True)
sig_arr = df[SIGNALS].ffill().bfill().fillna(0).values.astype("float32")

# Standardise per channel (mean/std from 2024 only, no leakage)
mask_2024 = (df["timestamp"].dt.year == 2024).values
chan_mean = sig_arr[mask_2024].mean(axis=0)
chan_std  = sig_arr[mask_2024].std(axis=0) + 1e-6
sig_arr_z = (sig_arr - chan_mean) / chan_std

# Build (n, window, 4) windows where window[t] = sig[t-window:t]
n = len(df)
print(f"  building {n} windows of shape ({WINDOW}, {len(SIGNALS)})...", flush=True)
windows = np.zeros((n, WINDOW, len(SIGNALS)), dtype="float32")
for t in range(WINDOW, n):
    windows[t] = sig_arr_z[t - WINDOW:t]
# First WINDOW rows have all-zero windows (will be NaN-marked later for LGBM)

# ──────────────────────────────────────────────────────────────────────
# 3. LSTM Autoencoder
# ──────────────────────────────────────────────────────────────────────
class LSTMAE(nn.Module):
    def __init__(self, n_chan=4, n_hid=N_HID, n_bott=N_BOTT, window=WINDOW):
        super().__init__()
        self.n_chan = n_chan; self.n_hid = n_hid; self.n_bott = n_bott; self.window = window
        self.enc_lstm = nn.LSTM(n_chan, n_hid, batch_first=True)
        self.enc_lin  = nn.Linear(n_hid, n_bott)
        self.dec_lin  = nn.Linear(n_bott, n_hid)
        self.dec_lstm = nn.LSTM(n_hid, n_hid, batch_first=True)
        self.dec_out  = nn.Linear(n_hid, n_chan)

    def encode(self, x):                      # x: (B, T, C)
        _, (h, _) = self.enc_lstm(x)          # h: (1, B, H)
        return self.enc_lin(h.squeeze(0))     # (B, bott)

    def forward(self, x):
        z = self.encode(x)                                                 # (B, bott)
        h0 = self.dec_lin(z).unsqueeze(0)                                   # (1, B, H)
        c0 = torch.zeros_like(h0)
        # repeat z over T as decoder input
        rep = z.unsqueeze(1).repeat(1, self.window, 1)                      # (B, T, bott)
        rep = self.dec_lin(rep)                                              # (B, T, H)
        out, _ = self.dec_lstm(rep, (h0, c0))                                # (B, T, H)
        return self.dec_out(out)                                             # (B, T, C)

# ──────────────────────────────────────────────────────────────────────
# 4. Train AE on 2024 windows only (no 2025 leakage)
# ──────────────────────────────────────────────────────────────────────
train_idx = np.where(mask_2024 & (np.arange(n) >= WINDOW))[0]
print(f"AE training rows: {len(train_idx)} (2024 only, post-WINDOW)", flush=True)

# Subsample for speed (every 4th row = hourly)
train_idx = train_idx[::4]
print(f"  subsampled to {len(train_idx)} (every 4th = hourly cadence)", flush=True)

X_train = torch.from_numpy(windows[train_idx]).float()    # (N, T, C)

torch.manual_seed(42)
model = LSTMAE().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCHS = 6
BATCH  = 256
print(f"Training LSTM-AE for {EPOCHS} epochs, batch={BATCH}, n={len(X_train)}...", flush=True)
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    losses = []
    for i in range(0, len(X_train), BATCH):
        bi = perm[i:i+BATCH]
        xb = X_train[bi].to(DEVICE)
        yh = model(xb)
        loss = loss_fn(yh, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {ep+1}/{EPOCHS}  recon_mse={np.mean(losses):.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 5. Encode all rows -> 16-dim bottleneck features
# ──────────────────────────────────────────────────────────────────────
print("Encoding all rows...", flush=True)
model.eval()
bott_all = np.zeros((n, N_BOTT), dtype="float32")
with torch.no_grad():
    for i in range(WINDOW, n, 512):
        xb = torch.from_numpy(windows[i:i+512]).float().to(DEVICE)
        z  = model.encode(xb).cpu().numpy()
        bott_all[i:i+len(z)] = z

# Add as features
ae_cols = [f"ae_{i:02d}" for i in range(N_BOTT)]
for i, c in enumerate(ae_cols):
    df[c] = bott_all[:, i]
df.loc[df.index < WINDOW, ae_cols] = np.nan
print(f"  added {N_BOTT} LSTM-AE bottleneck features", flush=True)

# ──────────────────────────────────────────────────────────────────────
# 6. 8-bag LGBM walkforward (same pipeline as 60.65% baseline)
# ──────────────────────────────────────────────────────────────────────
DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
             "qow","hod","net_load","sell_price"}
feats_all = [c for c in df.columns if c not in DROP_BASE]
print(f"v10 features: {len(feats_all)}  (v7 + {N_BOTT} LSTM-AE bottleneck)", flush=True)

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
    {"num_leaves":63, "max_depth":8, "learning_rate":0.02, "min_child_samples":20,
     "reg_alpha":0.1, "reg_lambda":0.1, "subsample":0.9, "colsample_bytree":0.9},
    {"num_leaves":47, "max_depth":7, "learning_rate":0.015,"min_child_samples":30,
     "reg_alpha":0.3, "reg_lambda":0.5, "subsample":0.85,"colsample_bytree":0.85},
    {"num_leaves":95, "max_depth":10,"learning_rate":0.025,"min_child_samples":15,
     "reg_alpha":0.05,"reg_lambda":0.1, "subsample":0.85,"colsample_bytree":0.85},
    {"num_leaves":31, "max_depth":6, "learning_rate":0.025,"min_child_samples":40,
     "reg_alpha":0.5, "reg_lambda":0.5, "subsample":0.85,"colsample_bytree":0.85},
]

def run(label, splits, n_bags=8, n_trees=1500):
    tr_m, va_m, te_m = splits
    tv_df = pd.concat([df[tr_m], df[va_m]], ignore_index=True).dropna(subset=feats_all).reset_index(drop=True)
    te_df = df[te_m].dropna(subset=feats_all).reset_index(drop=True)
    print(f"\n=== {label}  train+val={len(tv_df)}  test={len(te_df)} ===", flush=True)
    X_tv, y_tv = tv_df[feats_all].values, tv_df["load_kw"].values
    X_te, y_te = te_df[feats_all].values, te_df["load_kw"].values
    preds = np.zeros((n_bags, len(X_te)))
    importance = np.zeros(len(feats_all))
    t0 = time.time()
    for i in range(n_bags):
        cfg = dict(LIGHT_CONFIGS[i % len(LIGHT_CONFIGS)])
        seed = 42 + i
        cfg.update({"n_estimators":n_trees, "subsample_freq":1, "objective":"huber",
                    "alpha":0.9, "verbose":-1, "n_jobs":-1, "random_state":seed})
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_tv), size=int(len(X_tv)*0.9))
        m = lgb.LGBMRegressor(**cfg); m.fit(X_tv[idx], y_tv[idx])
        preds[i] = np.clip(m.predict(X_te), 0, None)
        importance += m.feature_importances_ / n_bags
        print(f"  bag {i+1}/{n_bags} ({time.time()-t0:.0f}s)", flush=True)
    avg = preds.mean(axis=0)
    n_te = nrmse(y_te, avg)
    print(f"  {label} NRMSE: {n_te:.2f}%", flush=True)
    # AE feature importance
    ae_imp = sorted([(f, importance[feats_all.index(f)]) for f in ae_cols], key=lambda x: -x[1])
    print("  AE feature importances:")
    for f, im in ae_imp[:5]: print(f"    {f}: {im:.0f}")
    return te_df["timestamp"].values, avg, y_te, n_te

ts_a, p_a, y_a, n_a = run("APRIL",     april_split())
ts_s, p_s, y_s, n_s = run("SEPTEMBER", sept_split())

ts_all = np.concatenate([ts_a, ts_s])
p_all  = np.concatenate([p_a, p_s])
y_all  = np.concatenate([y_a, y_s])
n_combined = nrmse(y_all, p_all)

print("\n" + "="*70, flush=True)
print(f"  v10 (LSTM-AE features) combined NRMSE: {n_combined:.2f}%", flush=True)
print(f"  vs prev best (online_retraining): 60.65%", flush=True)
delta = n_combined - 60.65
verdict = "WORSE" if delta > 0 else "IMPROVEMENT!" if delta < 0 else "tied"
print(f"  delta vs baseline: {delta:+.2f}pp  ({verdict})", flush=True)

out = ROOT / "outputs/forecasts/v10_lstm_ae_test_preds.csv"
pd.DataFrame({"timestamp": ts_all, "load_pred": p_all}).to_csv(out, index=False)
print(f"\nSaved -> {out}", flush=True)
