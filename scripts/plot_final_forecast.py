"""
Plot the final (winning) forecast vs ground truth for April + September 2025.

Defaults to outputs/forecasts/final_blend_test_preds.csv but can be overridden:
  python plot_final_forecast.py --pred outputs/forecasts/<file>.csv

Outputs:
  outputs/plots/forecasting/final_forecast_april.png
  outputs/plots/forecasting/final_forecast_september.png
  outputs/plots/forecasting/final_forecast_combined.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parents[1]
OUT  = ROOT / "outputs/plots/forecasting"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default=str(ROOT / "outputs/forecasts/final_blend_test_preds.csv"))
ap.add_argument("--label", default="final_blend")
args = ap.parse_args()

# Load actuals
df_v7 = pd.read_parquet(ROOT / "data/features/features_v7_all.parquet")
df_v7["timestamp"] = pd.to_datetime(df_v7["timestamp"])
test_mask = (df_v7["timestamp"].dt.year == 2025) & (df_v7["timestamp"].dt.month.isin([4,9]))
actuals = df_v7[test_mask][["timestamp","load_kw"]].sort_values("timestamp").reset_index(drop=True)

# Load predictions
preds = pd.read_csv(args.pred, parse_dates=["timestamp"])
merged = actuals.merge(preds, on="timestamp", how="left")
merged["load_pred"] = merged["load_pred"].ffill().bfill()

y = merged["load_kw"].values
p = merged["load_pred"].values
ts = merged["timestamp"]
def nrmse(yy, pp): return float(np.sqrt(np.mean((yy-pp)**2)) / np.mean(yy) * 100)
n_total = nrmse(y, p)
mask_apr = ts.dt.month == 4
mask_sep = ts.dt.month == 9
n_apr = nrmse(y[mask_apr], p[mask_apr])
n_sep = nrmse(y[mask_sep], p[mask_sep])
print(f"Loaded {len(merged)} rows. NRMSE: combined={n_total:.2f}%  Apr={n_apr:.2f}%  Sep={n_sep:.2f}%")

def plot_month(ax, mask, title, n_metric, color="tab:blue"):
    ts_m = ts[mask]; y_m = y[mask]; p_m = p[mask]
    ax.plot(ts_m, y_m, color="black", lw=0.8, alpha=0.85, label="Actual load")
    ax.plot(ts_m, p_m, color=color, lw=0.8, alpha=0.85, label=f"Forecast")
    ax.fill_between(ts_m, y_m, p_m, alpha=0.12, color=color)
    ax.set_ylabel("Load (kW)")
    ax.set_title(f"{title}  |  NRMSE = {n_metric:.2f}%", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, alpha=0.3)

# Per-month
for tag, mk, name, n_m in [
    ("april",     mask_apr, "April 2025",     n_apr),
    ("september", mask_sep, "September 2025", n_sep),
]:
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_month(ax, mk, name, n_m, color="tab:blue" if tag == "april" else "tab:orange")
    fig.suptitle(f"{args.label}: forecast vs actual — {name}", fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT / f"final_forecast_{tag}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

# Combined two-panel
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharey=False)
plot_month(axes[0], mask_apr, "April 2025",     n_apr, color="tab:blue")
plot_month(axes[1], mask_sep, "September 2025", n_sep, color="tab:orange")
fig.suptitle(f"{args.label}: forecast vs actual  (combined NRMSE = {n_total:.2f}%)", fontsize=13, y=1.00)
fig.tight_layout()
out = OUT / "final_forecast_combined.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")

# Optional: zoom into a representative week (first week of each month)
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharey=False)
zoom_apr = mask_apr & (ts.dt.day <= 7)
zoom_sep = mask_sep & (ts.dt.day <= 7)
plot_month(axes[0], zoom_apr, "April 2025  (Apr 1-7)",     nrmse(y[zoom_apr], p[zoom_apr]), color="tab:blue")
plot_month(axes[1], zoom_sep, "September 2025  (Sep 1-7)", nrmse(y[zoom_sep], p[zoom_sep]), color="tab:orange")
fig.suptitle(f"{args.label}: forecast vs actual — first-week zoom", fontsize=13, y=1.00)
fig.tight_layout()
out = OUT / "final_forecast_zoom_week1.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
