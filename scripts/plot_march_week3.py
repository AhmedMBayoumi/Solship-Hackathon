"""
March Week 3 dispatch plot — MANDATORY deliverable per Phase 6.1.
5-panel: load, PV, P_battery, P_grid, SoC vs time for March 16-22, 2025.

We have to RUN MPC over March 2025 first (since our test scope was Apr+Sep).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.controller.mpc_loop import run_mpc, SOC_INIT
from src.eval.compute_bill import compute_bill

ROOT = Path(__file__).parents[1]
H = int(sys.argv[1]) if len(sys.argv) > 1 else 192   # use the knee horizon

# Use full features so we have lag history for forecasting in March
df_all = pd.read_parquet(ROOT / "data/features/features_all.parquet")
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

# Filter to March 2025
df_mar = df_all[(df_all["timestamp"].dt.year == 2025) &
                (df_all["timestamp"].dt.month == 3)].copy().reset_index(drop=True)
print(f"March 2025: {len(df_mar)} rows")
print(f"  cols: {list(df_mar.columns[:10])}")

# Generate forecasts for March 2025 using the trained LightGBM model
import pickle
mp = pickle.load(open(ROOT / "outputs/models/lgbm_model.pkl", "rb"))
model = mp["model"]
feature_cols = mp["feature_cols"]
avail = [c for c in feature_cols if c in df_mar.columns]
print(f"  features available for inference: {len(avail)}")

# Pre-compute predictions causally (using actual lag features that are populated by build_features)
X = df_mar[avail].values
mask = ~np.any(np.isnan(X), axis=1)
preds = np.full(len(df_mar), np.nan)
preds[mask] = model.predict(X[mask])
# Fill NaN forward for early rows
for i in range(len(preds)):
    if np.isnan(preds[i]):
        preds[i] = preds[i-1] if i > 0 else 1.0

df_mar = df_mar.assign(load_pred=preds)


def factory(df_month):
    load_pred = df_month["load_pred"].values
    def fn(t, H_in):
        end = min(t + H_in, len(load_pred))
        return load_pred[t:end]
    return fn


print(f"Running MPC on March 2025 with H={H}...")
t0 = time.time()
res, bill = run_mpc(df_mar, factory(df_mar), H=H, soc_init=SOC_INIT, verbose=False)
elapsed = time.time() - t0
print(f"  Bill: EUR {bill['net_bill']:+.2f}  ({elapsed:.0f}s)")

# Filter to Week 3: March 16-22, 2025 (Mon-Sun)
WEEK_START = pd.Timestamp("2025-03-16")
WEEK_END   = pd.Timestamp("2025-03-22 23:59:59")
sub = res[(res["timestamp"] >= WEEK_START) & (res["timestamp"] <= WEEK_END)].copy()
print(f"  Week 3 slice: {len(sub)} timesteps")

# Tariff bands per timestep — re-derive
def tariff_band(ts):
    is_holiday = ts.weekday() == 6 or (ts.month, ts.day) in {(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)}
    h = ts.hour
    if is_holiday:               return "F3"
    if ts.weekday() == 5:        # Saturday
        return "F2" if 7 <= h < 23 else "F3"
    # Mon-Fri
    if 8 <= h < 19:              return "F1"
    if 7 <= h < 8 or 19 <= h < 23: return "F2"
    return "F3"

# Plot
fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
ts = sub["timestamp"]

# 1. Load
axes[0].plot(ts, sub["load_kw"], color="#0066cc", lw=1.0, label="Load")
axes[0].fill_between(ts, 0, sub["load_kw"], alpha=0.15, color="#0066cc")
axes[0].set_ylabel("Load\n(kW)")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title(f"March Week 3 (Mar 16-22, 2025) — MPC dispatch with H={H}, bill = EUR {bill['net_bill']:+.2f}",
                  fontsize=12)

# 2. PV
axes[1].plot(ts, sub["pv_kw"], color="orange", lw=1.0, label="PV")
axes[1].fill_between(ts, 0, sub["pv_kw"], alpha=0.2, color="orange")
axes[1].set_ylabel("PV\n(kW)")
axes[1].grid(alpha=0.3)
axes[1].legend(loc="upper right", fontsize=9)

# 3. Battery
axes[2].plot(ts, sub["p_battery_kw"], color="#9933cc", lw=1.0, label="Battery (+ discharge)")
axes[2].fill_between(ts, 0, sub["p_battery_kw"], where=sub["p_battery_kw"]>0, alpha=0.3, color="green", label="discharging")
axes[2].fill_between(ts, 0, sub["p_battery_kw"], where=sub["p_battery_kw"]<0, alpha=0.3, color="red", label="charging")
axes[2].axhline(0, color="black", lw=0.5)
axes[2].set_ylabel("P_battery\n(kW)")
axes[2].grid(alpha=0.3)
axes[2].legend(loc="upper right", fontsize=8)

# 4. Grid
axes[3].plot(ts, sub["p_grid_kw"], color="#cc3333", lw=1.0, label="Grid (+ import)")
axes[3].fill_between(ts, 0, sub["p_grid_kw"], where=sub["p_grid_kw"]>0, alpha=0.2, color="red", label="import")
axes[3].fill_between(ts, 0, sub["p_grid_kw"], where=sub["p_grid_kw"]<0, alpha=0.2, color="green", label="export")
axes[3].axhline(0, color="black", lw=0.5)
axes[3].set_ylabel("P_grid\n(kW)")
axes[3].grid(alpha=0.3)
axes[3].legend(loc="upper right", fontsize=8)

# 5. SoC
axes[4].plot(ts, sub["soc"]*100, color="#009999", lw=1.5, label="SoC (%)")
axes[4].fill_between(ts, 0, sub["soc"]*100, alpha=0.2, color="#009999")
axes[4].set_ylim(0, 100)
axes[4].set_ylabel("SoC\n(%)")
axes[4].grid(alpha=0.3)
axes[4].legend(loc="upper right", fontsize=9)

# Tariff band shading on top panel
band_colors = {"F1": "#ffd700", "F2": "#ff6347", "F3": "#90ee90"}
for i, t in enumerate(ts):
    band = tariff_band(t)
    for ax in axes:
        ax.axvspan(t, t + pd.Timedelta(minutes=15), alpha=0.05, color=band_colors[band], zorder=0)

axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %m-%d"))
axes[-1].set_xlabel("Date")

plt.tight_layout()
out_dir = ROOT / "outputs/plots"
out_dir.mkdir(exist_ok=True)
out = out_dir / "march_week3_dispatch.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved -> {out}")
