"""
Week-ahead forecast v3 — v2 (anchor + horizon + long-lag) PLUS LSTM-AE features.

Each prediction at time t = anchor + h uses:
  - Long-lag features at t (≥1 week back, all in training data)
  - Anchor-state features summarising 24h ending at the anchor (constant across forecast)
  - LSTM-AE bottleneck (16 dims) of past window ending at the anchor (constant)
  - Horizon h (steps from anchor)
  - Calendar features at t

The LSTM-AE encodes the recent multivariate history in 16 dims so the model
sees a richer "what did the household look like just before forecasting" signal.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

# torch FIRST (avoid OpenMP DLL collision with lightgbm)
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
SRC  = ROOT / "data/processed/dataset_processed.csv"
OUT  = ROOT / "day 2/submission 1"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"

print("Loading 2024+2025 dataset...", flush=True)
df_orig = pd.read_csv(SRC, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
print(f"  rows={len(df_orig)}", flush=True)

# Forecast timestamps
fc_ts = pd.date_range(start="2026-01-01 00:00:00", end="2026-01-07 23:45:00", freq="15min")
fc_df = pd.DataFrame({"timestamp": fc_ts, "load_kw": np.nan, "pv_kw": np.nan,
                      "p_battery_kw": np.nan, "grid_kw": np.nan})
df_all = pd.concat([df_orig[["timestamp","load_kw","pv_kw","p_battery_kw","grid_kw"]], fc_df],
                   ignore_index=True).sort_values("timestamp").reset_index(drop=True)
ts = df_all["timestamp"]
n  = len(df_all)

# ── Calendar / cyclical / holiday ───────────────────────────────────
df_all["hour"]         = ts.dt.hour
df_all["dow"]          = ts.dt.dayofweek
df_all["month"]        = ts.dt.month
df_all["day_of_year"]  = ts.dt.dayofyear
df_all["week_of_year"] = ts.dt.isocalendar().week.astype(int)
df_all["is_weekend"]   = (df_all["dow"]>=5).astype(int)
df_all["minute"]       = ts.dt.minute
df_all["hod"]          = df_all["hour"]*4 + df_all["minute"]//4

IT_HOLS = pd.to_datetime([
    "2024-01-01","2024-01-06","2024-04-01","2024-04-25","2024-05-01",
    "2024-06-02","2024-08-15","2024-11-01","2024-12-08","2024-12-25","2024-12-26",
    "2025-01-01","2025-01-06","2025-04-21","2025-04-25","2025-05-01",
    "2025-06-02","2025-08-15","2025-11-01","2025-12-08","2025-12-25","2025-12-26",
    "2026-01-01","2026-01-06",
])
df_all["is_holiday"] = ts.dt.normalize().isin(IT_HOLS).astype(int)
hd = ts.dt.normalize()
df_all["is_bridge_day"] = (
    (hd.shift(-96).isin(IT_HOLS) & ~df_all["is_holiday"].astype(bool) & ~df_all["is_weekend"].astype(bool)) |
    (hd.shift(96).isin(IT_HOLS)  & ~df_all["is_holiday"].astype(bool) & ~df_all["is_weekend"].astype(bool))
).astype(int)
for period, label in [(96,"24h"),(48,"12h"),(32,"8h"),(24,"6h"),(16,"4h")]:
    df_all[f"sin_{label}"] = np.sin(2*np.pi*df_all["hod"]/period)
    df_all[f"cos_{label}"] = np.cos(2*np.pi*df_all["hod"]/period)
df_all["sin_annual"]  = np.sin(2*np.pi*df_all["day_of_year"]/365.25)
df_all["cos_annual"]  = np.cos(2*np.pi*df_all["day_of_year"]/365.25)
df_all["it_morning_rush"] = ((df_all["hour"]>=7) & (df_all["hour"]<9)).astype(int)
df_all["it_lunch_peak"]   = ((df_all["hour"]>=12) & (df_all["hour"]<14)).astype(int)
df_all["it_dinner_hour"]  = ((df_all["hour"]>=19) & (df_all["hour"]<22)).astype(int)
df_all["it_pre_dawn"]     = ((df_all["hour"]>=1) & (df_all["hour"]<5)).astype(int)

# ── Long-lags ─────────────────────────────────────────────────────────
LONG_LAGS = [672, 768, 864, 960, 1344, 2016, 4032, 8064]
for lag in LONG_LAGS:
    df_all[f"lag_{lag}"]        = df_all["load_kw"].shift(lag)
    df_all[f"pv_lag_{lag}"]     = df_all["pv_kw"].shift(lag)
    df_all[f"battery_lag_{lag}"]= df_all["p_battery_kw"].shift(lag)
    df_all[f"grid_lag_{lag}"]   = df_all["grid_kw"].shift(lag)
df_all["d_week"]  = df_all["lag_672"]  - df_all["lag_1344"]
df_all["d_2week"] = df_all["lag_1344"] - df_all["lag_2016"]

# ── Trailing-24h state (used as anchor features) ──────────────────────
df_all["trail24_load_mean"] = df_all["load_kw"].rolling(96, min_periods=10).mean()
df_all["trail24_load_std"]  = df_all["load_kw"].rolling(96, min_periods=10).std()
df_all["trail24_load_max"]  = df_all["load_kw"].rolling(96, min_periods=10).max()
df_all["trail24_load_min"]  = df_all["load_kw"].rolling(96, min_periods=10).min()
df_all["trail3d_load_mean"] = df_all["load_kw"].rolling(288, min_periods=10).mean()
df_all["trail7d_load_mean"] = df_all["load_kw"].rolling(672, min_periods=10).mean()
df_all["trail24_pv_mean"]   = df_all["pv_kw"].rolling(96, min_periods=10).mean()
df_all["trail24_pv_max"]    = df_all["pv_kw"].rolling(96, min_periods=10).max()
df_all["trail24_battery_mean"]= df_all["p_battery_kw"].rolling(96, min_periods=10).mean()
df_all["trail24_grid_mean"] = df_all["grid_kw"].rolling(96, min_periods=10).mean()
df_all["last_load"]         = df_all["load_kw"].shift(0)
df_all["last_pv"]           = df_all["pv_kw"].shift(0)

ANCHOR_FEATS = [
    "trail24_load_mean","trail24_load_std","trail24_load_max","trail24_load_min",
    "trail3d_load_mean","trail7d_load_mean",
    "trail24_pv_mean","trail24_pv_max","trail24_battery_mean","trail24_grid_mean",
    "last_load","last_pv",
]

# ── LSTM-AE: train on past windows of [load, pv, battery, grid] ───────
WINDOW, N_BOTT, N_HID = 96, 16, 32
SIGNALS = ["load_kw", "pv_kw", "p_battery_kw", "grid_kw"]
print(f"\nBuilding past-{WINDOW} windows of {SIGNALS}...", flush=True)
sig_arr = df_all[SIGNALS].fillna(0).values.astype("float32")
# Standardise on training data only (rows with valid load_kw, i.e. non-2026)
train_mask = df_all["load_kw"].notna().values
chan_mean = sig_arr[train_mask].mean(axis=0)
chan_std  = sig_arr[train_mask].std(axis=0) + 1e-6
sig_z = (sig_arr - chan_mean) / chan_std

windows = np.zeros((n, WINDOW, len(SIGNALS)), dtype="float32")
for t in range(WINDOW, n):
    windows[t] = sig_z[t-WINDOW:t]

class LSTMAE(nn.Module):
    def __init__(self, n_chan=4, n_hid=N_HID, n_bott=N_BOTT, window=WINDOW):
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

# Train on train rows hourly subsample
ae_train_idx = np.where(train_mask & (np.arange(n) >= WINDOW))[0][::4]
X_ae = torch.from_numpy(windows[ae_train_idx]).float()
print(f"  AE training rows: {len(X_ae)}", flush=True)
torch.manual_seed(42)
model = LSTMAE(n_chan=len(SIGNALS)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.MSELoss()
EPOCHS, BATCH = 12, 256
print(f"Training LSTM-AE: {EPOCHS} epochs, batch={BATCH}", flush=True)
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(len(X_ae))
    losses = []
    for i in range(0, len(X_ae), BATCH):
        xb = X_ae[perm[i:i+BATCH]].to(DEVICE)
        yh = model(xb); loss = loss_fn(yh, xb)
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
    df_all[c] = bott[:, i]

ANCHOR_FEATS = ANCHOR_FEATS + ae_cols   # AE bottleneck used as anchor features too

# ── Build training samples (random horizon ↔ anchor pairs) ───────────
NON_ANCHOR_COLS = [c for c in df_all.columns
                   if c not in {"timestamp","load_kw","pv_kw","p_battery_kw","grid_kw","minute","hod"}
                   and c not in ANCHOR_FEATS]

target_idx = df_all.index[df_all["load_kw"].notna() & df_all["lag_8064"].notna()]
target_idx = target_idx[target_idx >= 8064]
print(f"\neligible target rows: {len(target_idx)}", flush=True)

rng = np.random.default_rng(42)
horizons = rng.integers(1, 673, size=len(target_idx))
df_target  = df_all.loc[target_idx, NON_ANCHOR_COLS + ["load_kw"]].reset_index(drop=True)
df_target["horizon"] = horizons
anchor_indices = target_idx - horizons
df_anchor = df_all.loc[anchor_indices, ANCHOR_FEATS].reset_index(drop=True)
df_anchor.columns = [f"anchor_{c}" for c in df_anchor.columns]
df_train = pd.concat([df_target, df_anchor], axis=1).dropna().reset_index(drop=True)
feats = [c for c in df_train.columns if c != "load_kw"]
X_tr  = df_train[feats].values
y_tr  = df_train["load_kw"].values
print(f"  train shape: {X_tr.shape}", flush=True)

# ── Forecast samples ─────────────────────────────────────────────────
ANCHOR_TS = df_orig["timestamp"].iloc[-1]
anchor_idx = df_all.index[ts == ANCHOR_TS][0]
print(f"\nAnchor = {ANCHOR_TS}  (idx={anchor_idx})", flush=True)
fc_idx = df_all.index[(ts.dt.year == 2026) & (ts.dt.month == 1) & (ts.dt.day <= 7)]

df_fc = df_all.loc[fc_idx, NON_ANCHOR_COLS].reset_index(drop=True)
df_fc["horizon"] = (fc_idx - anchor_idx).values
anchor_state = df_all.loc[anchor_idx, ANCHOR_FEATS].to_dict()
for c, v in anchor_state.items():
    df_fc[f"anchor_{c}"] = v
df_fc = df_fc[feats].copy()
df_fc = df_fc.ffill().bfill()
X_fc = df_fc.values

# ── 8-bag LGBM ───────────────────────────────────────────────────────
LIGHT_CONFIGS = [
    {"num_leaves":63,"max_depth":8,"learning_rate":0.025,"min_child_samples":25,
     "reg_alpha":0.1,"reg_lambda":0.1,"subsample":0.9,"colsample_bytree":0.9},
    {"num_leaves":47,"max_depth":7,"learning_rate":0.02,"min_child_samples":40,
     "reg_alpha":0.3,"reg_lambda":0.5,"subsample":0.85,"colsample_bytree":0.85},
    {"num_leaves":95,"max_depth":10,"learning_rate":0.03,"min_child_samples":20,
     "reg_alpha":0.05,"reg_lambda":0.1,"subsample":0.85,"colsample_bytree":0.85},
    {"num_leaves":31,"max_depth":6,"learning_rate":0.03,"min_child_samples":50,
     "reg_alpha":0.5,"reg_lambda":0.5,"subsample":0.85,"colsample_bytree":0.85},
]
n_bags = 8
preds = np.zeros((n_bags, len(X_fc)))
print(f"\n=== Training {n_bags} bags  train={len(X_tr)}  fc={len(X_fc)} ===", flush=True)
t0 = time.time()
for i in range(n_bags):
    cfg = dict(LIGHT_CONFIGS[i % len(LIGHT_CONFIGS)])
    seed = 42 + i
    cfg.update({"n_estimators":1500,"subsample_freq":1,"objective":"huber",
                "alpha":0.9,"verbose":-1,"n_jobs":-1,"random_state":seed})
    rng2 = np.random.default_rng(seed)
    idx = rng2.integers(0, len(X_tr), size=int(len(X_tr)*0.9))
    m = lgb.LGBMRegressor(**cfg); m.fit(X_tr[idx], y_tr[idx])
    preds[i] = np.clip(m.predict(X_fc), 0, None)
    print(f"  bag {i+1}/{n_bags}  ({time.time()-t0:.0f}s)", flush=True)
avg = preds.mean(axis=0)

# Post-processing
p_sm = pd.Series(avg).rolling(3, min_periods=1, center=True).mean().values
alpha_vp = avg.std() / max(p_sm.std(), 1e-9)
mean_sm = p_sm.mean()
p_final = np.clip(mean_sm + alpha_vp * (p_sm - mean_sm), 0, None)
print(f"\nPost-processing: MA(3) + alpha={alpha_vp:.3f}", flush=True)

df_out = pd.DataFrame({
    "Timestamps": df_all.loc[fc_idx, "timestamp"].values,
    "Load":       p_final,
})
out_csv = OUT / "WeekAhead_Jan1to7_2026.csv"
df_out.to_csv(out_csv, index=False)
print(f"Saved CSV -> {out_csv}", flush=True)
print(f"\nForecast stats:")
print(f"  mean: {p_final.mean():.3f}  std: {p_final.std():.3f}  min: {p_final.min():.3f}  max: {p_final.max():.3f}")
