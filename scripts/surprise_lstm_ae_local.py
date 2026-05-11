"""LOCAL: LSTM-AE + 8-bag LGBM on surprise dataset. Predict March 2026.

Same pipeline as v10 but for the surprise data (load_p, pv_p only).
Torch CPU works locally if imported BEFORE lightgbm.
"""
import sys, time, warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
DEVICE = "cpu"
print(f"Device: {DEVICE}", flush=True)

print("Loading surprise features...", flush=True)
df = pd.read_parquet(ROOT / "data/features/features_surprise_all.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True).ffill().bfill()
print(f"  rows={len(df)}  cols={df.shape[1]}", flush=True)

WINDOW, N_BOTT, N_HID = 96, 16, 32
SIGNALS = ["load_kw", "pv_kw"]   # surprise dataset has only these two raw signals

# Standardise on training data only (exclude March 2026)
train_mask = ~((df["timestamp"].dt.year == 2026) & (df["timestamp"].dt.month == 3))
train_mask = train_mask.values
sig_arr = df[SIGNALS].values.astype("float32")
chan_mean = sig_arr[train_mask].mean(axis=0)
chan_std  = sig_arr[train_mask].std(axis=0) + 1e-6
sig_z = (sig_arr - chan_mean) / chan_std

n = len(df)
print(f"Building {n} past-{WINDOW} windows of {SIGNALS}...", flush=True)
windows = np.zeros((n, WINDOW, len(SIGNALS)), dtype="float32")
for t in range(WINDOW, n):
    windows[t] = sig_z[t-WINDOW:t]

class LSTMAE(nn.Module):
    def __init__(self, n_chan=2, n_hid=N_HID, n_bott=N_BOTT, window=WINDOW):
        super().__init__()
        self.window = window
        self.enc_lstm = nn.LSTM(n_chan, n_hid, batch_first=True)
        self.enc_lin  = nn.Linear(n_hid, n_bott)
        self.dec_lin  = nn.Linear(n_bott, n_hid)
        self.dec_lstm = nn.LSTM(n_hid, n_hid, batch_first=True)
        self.dec_out  = nn.Linear(n_hid, n_chan)
    def encode(self, x):
        _, (h, _) = self.enc_lstm(x); return self.enc_lin(h.squeeze(0))
    def forward(self, x):
        z = self.encode(x); rep = z.unsqueeze(1).repeat(1, self.window, 1)
        rep = self.dec_lin(rep); h0 = self.dec_lin(z).unsqueeze(0)
        c0 = torch.zeros_like(h0); out, _ = self.dec_lstm(rep, (h0, c0))
        return self.dec_out(out)

train_idx = np.where(train_mask & (np.arange(n) >= WINDOW))[0][::4]
X_train = torch.from_numpy(windows[train_idx]).float()
print(f"AE training rows: {len(X_train)}", flush=True)

torch.manual_seed(42)
model = LSTMAE(n_chan=len(SIGNALS)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCHS, BATCH = 12, 256
print(f"Training LSTM-AE: {EPOCHS} epochs, batch={BATCH}", flush=True)
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    losses = []
    for i in range(0, len(X_train), BATCH):
        xb = X_train[perm[i:i+BATCH]].to(DEVICE)
        yh = model(xb)
        loss = loss_fn(yh, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {ep+1}/{EPOCHS}  recon_mse={np.mean(losses):.4f}  ({time.time()-t0:.0f}s)", flush=True)

print("Encoding all rows...", flush=True)
model.eval()
bott = np.zeros((n, N_BOTT), dtype="float32")
with torch.no_grad():
    for i in range(WINDOW, n, 512):
        xb = torch.from_numpy(windows[i:i+512]).float().to(DEVICE)
        bott[i:i+len(xb)] = model.encode(xb).cpu().numpy()
ae_cols = [f"ae_{i:02d}" for i in range(N_BOTT)]
for i, c in enumerate(ae_cols):
    df[c] = bott[:, i]
df.loc[df.index < WINDOW, ae_cols] = np.nan
df = df.ffill().bfill()
print(f"  added {N_BOTT} AE features", flush=True)

# 8-bag LGBM
DROP = {"timestamp","load_kw","pv_kw","minute","qow","hod","net_load"}
feats = [c for c in df.columns if c not in DROP]
print(f"Features: {len(feats)}", flush=True)

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

test_mask = ((df["timestamp"].dt.year == 2026) & (df["timestamp"].dt.month == 3)).values
train_full = ~test_mask
df_tv = df[train_full].dropna(subset=feats).reset_index(drop=True)
df_te = df[test_mask].dropna(subset=feats).reset_index(drop=True)
X_tv, y_tv = df_tv[feats].values, df_tv["load_kw"].values
X_te, y_te = df_te[feats].values, df_te["load_kw"].values
n_bags, n_trees = 8, 1500
preds = np.zeros((n_bags, len(X_te)))
print(f"\n=== Training {n_bags} bags  train={len(X_tv)} test={len(X_te)} ===", flush=True)
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
    print(f"  bag {i+1}/{n_bags} ({time.time()-t0:.0f}s)", flush=True)
avg = preds.mean(axis=0)

def rmse(y, p):  return float(np.sqrt(np.mean((y-p)**2)))
def mae(y, p):   return float(np.mean(np.abs(y-p)))
def mape(y, p):  return float(np.mean(np.abs(y-p) / np.maximum(np.abs(y), 0.01)) * 100)
def nrmse(y, p): return float(np.sqrt(np.mean((y-p)**2)) / np.mean(y) * 100)

print(f"\n=== LOCAL LSTM-AE + 8-bag LGBM (March 2026) ===")
print(f"  RMSE  : {rmse(y_te, avg):.4f} kW")
print(f"  MAE   : {mae(y_te, avg):.4f} kW")
print(f"  MAPE  : {mape(y_te, avg):.2f} %")
print(f"  NRMSE : {nrmse(y_te, avg):.2f} %")

out = ROOT / "outputs/forecasts/surprise_lstm_ae_local_test_preds.csv"
pd.DataFrame({"timestamp": df_te["timestamp"].values, "load_pred": avg}).to_csv(out, index=False)
print(f"Saved -> {out}")
