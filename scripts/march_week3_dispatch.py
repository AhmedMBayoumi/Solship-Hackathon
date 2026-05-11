"""
Generate the mandatory March Week 3 (Mar 17-23, 2025) dispatch plot.

We don't have a pre-existing forecast for March (our blend was for Apr+Sep),
so:
  1. Train a single 8-bag LGBM on v7 features using ALL data BEFORE March 17, 2025
     (no leakage from the test week itself)
  2. Forecast load_kw for Mar 17-23, 2025
  3. Run rolling-horizon MPC at H=96 with that causal forecast
  4. Plot load, PV, P_battery, P_grid, SoC across the week
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.controller.mpc_loop import run_mpc
from src.eval.compute_bill import compute_bill

ROOT = Path(__file__).parents[1]

# ── Load v7 features for the forecast ────────────────────────────────
print("Loading v7 features...", flush=True)
df = pd.read_parquet(ROOT / "data/features/features_v7_all.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True).ffill().bfill()
print(f"  rows={len(df)}  cols={df.shape[1]}", flush=True)

DROP = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
        "qow","hod","net_load","sell_price"}
feats = [c for c in df.columns if c not in DROP]
print(f"  features: {len(feats)}", flush=True)

# ── Train/test split: train < Mar 17 2025; test = Mar 17-23 2025 ─────
WEEK_START = pd.Timestamp("2025-03-17 00:00:00")
WEEK_END   = pd.Timestamp("2025-03-23 23:45:00")

train_mask = df["timestamp"] < WEEK_START
test_mask  = (df["timestamp"] >= WEEK_START) & (df["timestamp"] <= WEEK_END)

df_tr = df[train_mask].dropna(subset=feats).reset_index(drop=True)
df_te = df[test_mask].dropna(subset=feats).reset_index(drop=True)
print(f"  train rows: {len(df_tr)}  test rows: {len(df_te)}", flush=True)

X_tr, y_tr = df_tr[feats].values, df_tr["load_kw"].values
X_te, y_te = df_te[feats].values, df_te["load_kw"].values

# ── 8-bag LGBM ────────────────────────────────────────────────────────
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
n_bags = 8
preds = np.zeros((n_bags, len(X_te)))
print(f"\n=== Training {n_bags} bags ===", flush=True)
t0 = time.time()
for i in range(n_bags):
    cfg = dict(LIGHT_CONFIGS[i % len(LIGHT_CONFIGS)])
    seed = 42 + i
    cfg.update({"n_estimators":1500,"subsample_freq":1,"objective":"huber",
                "alpha":0.9,"verbose":-1,"n_jobs":-1,"random_state":seed})
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_tr), size=int(len(X_tr)*0.9))
    m = lgb.LGBMRegressor(**cfg); m.fit(X_tr[idx], y_tr[idx])
    preds[i] = np.clip(m.predict(X_te), 0, None)
    print(f"  bag {i+1}/{n_bags}  ({time.time()-t0:.0f}s)", flush=True)
avg = preds.mean(axis=0)
def nrmse(y, p): return float(np.sqrt(np.mean((y-p)**2)) / np.mean(y) * 100)
print(f"  March W3 forecast NRMSE: {nrmse(y_te, avg):.2f}%", flush=True)

# Save predictions for the MPC factory
preds_df = pd.DataFrame({
    "timestamp": df_te["timestamp"].values,
    "load_pred": avg,
})

# ── Run MPC at H=96 on March W3 ──────────────────────────────────────
print("\nRunning MPC on March W3 with our forecast...", flush=True)
df_2025 = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df_2025[df_2025["timestamp"].dt.year == 2025].copy().reset_index(drop=True)
df_w3 = df_2025[(df_2025["timestamp"] >= WEEK_START) & (df_2025["timestamp"] <= WEEK_END)].reset_index(drop=True)
print(f"  W3 raw rows: {len(df_w3)}", flush=True)

# Build the forecast factory
def make_factory(preds_df):
    def factory(df_month):
        merged = df_month[["timestamp"]].merge(preds_df, on="timestamp", how="left")
        load_pred = merged["load_pred"].values.astype(float)
        for i in range(len(load_pred)):
            if np.isnan(load_pred[i]):
                load_pred[i] = load_pred[i-1] if i > 0 else 1.0
        def fn(t, H_in):
            end = min(t + H_in, len(load_pred))
            return load_pred[t:end]
        return fn
    return factory

t0 = time.time()
res, bill = run_mpc(df_w3, make_factory(preds_df)(df_w3), H=96, verbose=False, label="MarW3")
print(f"  MPC done in {time.time()-t0:.0f}s", flush=True)
print(f"  March W3 bill: €{bill['net_bill']:+.2f}", flush=True)

# ── Plot 5 series ────────────────────────────────────────────────────
print("\nPlotting...", flush=True)
fig, axes = plt.subplots(5, 1, figsize=(15, 10), sharex=True)
ts_w3 = res["timestamp"]

axes[0].plot(ts_w3, res["load_kw"], color="black", lw=1.0, label="Actual load")
axes[0].plot(ts_w3, avg[:len(res)] if len(avg)==len(res) else preds_df["load_pred"].values,
             color="tab:blue", lw=0.9, alpha=0.7, label="Forecast")
axes[0].set_ylabel("Load (kW)"); axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title(f"March Week 3 (Mar 17-23, 2025) — MPC dispatch  |  Bill = €{bill['net_bill']:+.2f}",
                  fontsize=13, pad=8)

axes[1].plot(ts_w3, res["pv_kw"], color="tab:orange", lw=1.0)
axes[1].set_ylabel("PV (kW)"); axes[1].grid(True, alpha=0.3)

axes[2].plot(ts_w3, res["p_battery_kw"], color="tab:purple", lw=1.0)
axes[2].fill_between(ts_w3,
                     np.where(res["p_battery_kw"]>0, res["p_battery_kw"], 0),
                     0, color="tab:purple", alpha=0.20, label="discharge >0")
axes[2].fill_between(ts_w3,
                     np.where(res["p_battery_kw"]<0, res["p_battery_kw"], 0),
                     0, color="tab:cyan", alpha=0.20, label="charge <0")
axes[2].axhline(0, color="black", lw=0.4)
axes[2].axhline(+8, color="grey", lw=0.4, ls="--"); axes[2].axhline(-8, color="grey", lw=0.4, ls="--")
axes[2].set_ylabel("P_battery (kW)"); axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="upper right", fontsize=9)

axes[3].plot(ts_w3, res["p_grid_kw"], color="tab:red", lw=1.0)
axes[3].fill_between(ts_w3,
                     np.where(res["p_grid_kw"]>0, res["p_grid_kw"], 0),
                     0, color="tab:red", alpha=0.20, label="import >0")
axes[3].fill_between(ts_w3,
                     np.where(res["p_grid_kw"]<0, res["p_grid_kw"], 0),
                     0, color="tab:green", alpha=0.20, label="export <0")
axes[3].axhline(0, color="black", lw=0.4)
axes[3].axhline(+6, color="grey", lw=0.4, ls="--"); axes[3].axhline(-6, color="grey", lw=0.4, ls="--")
axes[3].set_ylabel("P_grid (kW)"); axes[3].grid(True, alpha=0.3)
axes[3].legend(loc="upper right", fontsize=9)

axes[4].plot(ts_w3, res["soc"]*100, color="tab:green", lw=1.0)
axes[4].fill_between(ts_w3, res["soc"]*100, 0, color="tab:green", alpha=0.10)
axes[4].set_ylabel("SoC (%)"); axes[4].set_ylim(0, 100)
axes[4].grid(True, alpha=0.3); axes[4].set_xlabel("Date")
axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
fig.tight_layout()

out_dir = ROOT / "outputs/plots/presentation"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "march_week3_dispatch.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"\nSaved -> {out}")

# Quick sanity numbers
print(f"\n=== Physics sanity checks ===")
print(f"  load mean: {res['load_kw'].mean():.2f}  max: {res['load_kw'].max():.2f}")
print(f"  pv mean:   {res['pv_kw'].mean():.2f}  max: {res['pv_kw'].max():.2f}")
print(f"  battery range: [{res['p_battery_kw'].min():.2f}, {res['p_battery_kw'].max():.2f}]  (limit ±8)")
print(f"  grid    range: [{res['p_grid_kw'].min():.2f}, {res['p_grid_kw'].max():.2f}]  (limit ±6)")
print(f"  SoC     range: [{res['soc'].min():.3f}, {res['soc'].max():.3f}]  (must be 0..1)")
print(f"  bill: €{bill['net_bill']:+.2f}")
