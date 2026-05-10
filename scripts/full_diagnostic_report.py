"""
Comprehensive diagnostic/analysis pipeline for our best forecast.
Outputs to outputs/plots/forecasting/ + outputs/reports/diagnostic_summary.txt:

  Plots:
    35_pred_vs_actual_FINAL.png         — scatter for current best
    36_residual_decomposition.png       — residual vs hour, dow, load level, PV
    37_per_day_nrmse_calendar.png       — heatmap of daily NRMSE
    38_largest_errors.png               — top-20 worst predictions in time
    39_error_by_context.png             — error vs lag_1, recent volatility, weather
    40_underprediction_zones.png        — when do we under-predict spikes?

  Report:
    diagnostic_summary.txt              — written analysis of failure modes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parents[1]
OUT_PLOTS = ROOT / "outputs/plots/forecasting"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_REPORT = ROOT / "outputs/reports/diagnostic_summary.txt"

# ── Load data ──────────────────────────────────────────────
preds_files = {
    "online_retraining_60.65":   "online_retraining_test_preds.csv",
    "rolling_weekly_60.75":      "rolling_weekly_test_preds.csv",
    "bagging_lgbm24_60.80":      "bagging_walkforward_lgbm24_test_preds.csv",
    "v5_PHANN_60.82":            "bagging_walkforward_v5_FINAL_test_preds.csv",
}
features = pd.read_parquet(ROOT / "data/features/features_v6_all.parquet")
features["timestamp"] = pd.to_datetime(features["timestamp"])
features_test = features[(features["timestamp"].dt.year == 2025) &
                         (features["timestamp"].dt.month.isin([4, 9]))].copy()

# Pick BEST forecast for the diagnostic (online_retraining at 60.65%)
preds = pd.read_csv(ROOT / "outputs/forecasts/online_retraining_test_preds.csv", parse_dates=["timestamp"])
m = features_test[["timestamp", "load_kw", "pv_kw", "lag_1", "lag_96",
                   "temperature_2m", "shortwave_radiation",
                   "is_weekend", "is_holiday", "tariff_enc",
                   "roll_4_std", "d_lag1", "is_high_state"]].merge(
    preds, on="timestamp", how="left"
).sort_values("timestamp").reset_index(drop=True)
m["resid"]   = m["load_kw"] - m["load_pred"]
m["abs_err"] = m["resid"].abs()
m["hour"]    = m["timestamp"].dt.hour
m["dow"]     = m["timestamp"].dt.dayofweek
m["month"]   = m["timestamp"].dt.month
m["day"]     = m["timestamp"].dt.normalize()

def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)
def rmse(y, yp):  return float(np.sqrt(np.mean((y-yp)**2)))

print(f"BEST forecast: online_retraining   NRMSE = {nrmse(m['load_kw'], m['load_pred']):.2f}%")

# ── Plot 35: Pred vs actual scatter ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (mo, name) in zip(axes, [(4, "April"), (9, "September")]):
    sub = m[m["month"] == mo]
    ax.scatter(sub["load_pred"], sub["load_kw"], s=8, alpha=0.25, c=sub["hour"], cmap="twilight")
    mx = max(sub["load_kw"].max(), sub["load_pred"].max()) * 1.05
    ax.plot([0, mx], [0, mx], "r--", lw=1.5)
    ax.set_xlabel("Predicted (kW)"); ax.set_ylabel("Actual (kW)")
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_title(f"{name} — NRMSE = {nrmse(sub['load_kw'], sub['load_pred']):.2f}%, RMSE = {rmse(sub['load_kw'], sub['load_pred']):.4f}")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOTS / "35_pred_vs_actual_FINAL.png", dpi=140, bbox_inches="tight")
plt.close()

# ── Plot 36: Residual decomposition ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# By hour
hour_g = m.groupby("hour").agg(rmse=("resid", lambda x: np.sqrt((x**2).mean())),
                                bias=("resid", "mean"), n=("resid", "count"),
                                actual_mean=("load_kw", "mean"))
hour_g["nrmse"] = hour_g["rmse"] / hour_g["actual_mean"] * 100
ax = axes[0,0]
ax.bar(hour_g.index, hour_g["rmse"], color="#0066cc", alpha=0.7, label="RMSE")
ax.set_xlabel("Hour of day"); ax.set_ylabel("RMSE (kW)")
ax.set_title("RMSE by hour-of-day")
ax2 = ax.twinx()
ax2.plot(hour_g.index, hour_g["bias"], "o-", color="red", label="bias")
ax2.axhline(0, color="black", lw=0.5)
ax2.set_ylabel("Bias (kW)")
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.3)

# By dow
dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_g = m.groupby("dow").agg(rmse=("resid", lambda x: np.sqrt((x**2).mean())),
                              bias=("resid", "mean"), actual_mean=("load_kw","mean"))
dow_g["nrmse"] = dow_g["rmse"] / dow_g["actual_mean"] * 100
ax = axes[0,1]
ax.bar(range(7), dow_g["nrmse"], color="#9933cc")
ax.set_xticks(range(7)); ax.set_xticklabels(dow_names)
ax.set_ylabel("NRMSE (%)"); ax.set_title("NRMSE by day-of-week")
for i, v in enumerate(dow_g["nrmse"]):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
ax.grid(alpha=0.3)

# By load level (Heteroscedasticity)
m["load_bin"] = pd.cut(m["load_kw"], bins=[-0.01, 0.5, 1.0, 1.5, 2.0, 3.0, 100],
                        labels=["0-0.5","0.5-1","1-1.5","1.5-2","2-3","3+"])
load_g = m.groupby("load_bin", observed=True).agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    bias=("resid","mean"), n=("resid","count"))
ax = axes[1,0]
ax.bar(range(len(load_g)), load_g["rmse"], color="#cc3333")
ax.set_xticks(range(len(load_g))); ax.set_xticklabels(load_g.index)
ax.set_xlabel("Actual load bin (kW)"); ax.set_ylabel("RMSE (kW)")
ax.set_title("RMSE grows with load level (heteroscedasticity)")
for i, (v, n) in enumerate(zip(load_g["rmse"], load_g["n"])):
    ax.text(i, v + 0.02, f"n={n}", ha="center", fontsize=8)
ax.grid(alpha=0.3)

# By PV level
m["pv_bin"] = pd.cut(m["pv_kw"], bins=[-0.01, 0.1, 1, 3, 5, 100], labels=["0","0-1","1-3","3-5","5+"])
pv_g = m.groupby("pv_bin", observed=True).agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    actual_mean=("load_kw","mean"))
pv_g["nrmse"] = pv_g["rmse"] / pv_g["actual_mean"] * 100
ax = axes[1,1]
ax.bar(range(len(pv_g)), pv_g["nrmse"], color="#cc6600")
ax.set_xticks(range(len(pv_g))); ax.set_xticklabels(pv_g.index)
ax.set_xlabel("PV bin (kW)"); ax.set_ylabel("NRMSE (%)")
ax.set_title("NRMSE vs PV generation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOTS / "36_residual_decomposition.png", dpi=140, bbox_inches="tight")
plt.close()

# ── Plot 37: Per-day NRMSE calendar ──────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
for ax, (mo, name) in zip(axes, [(4, "April"), (9, "September")]):
    sub = m[m["month"] == mo]
    daily = sub.groupby("day").agg(
        rmse=("resid", lambda x: np.sqrt((x**2).mean())),
        actual_mean=("load_kw","mean"))
    daily["nrmse"] = daily["rmse"] / daily["actual_mean"] * 100
    ax.bar(daily.index, daily["nrmse"], color="#0066cc")
    ax.axhline(daily["nrmse"].mean(), color="red", ls="--", label=f"avg={daily['nrmse'].mean():.1f}%")
    ax.set_title(f"{name} 2025 — daily NRMSE (best/worst days)")
    ax.set_ylabel("Daily NRMSE (%)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOTS / "37_per_day_nrmse_calendar.png", dpi=140, bbox_inches="tight")
plt.close()

# ── Plot 38: Top 20 worst predictions in time-series view ─
worst = m.nlargest(20, "abs_err")
print(f"\nTop 20 worst predictions:")
for _, r in worst.iterrows():
    print(f"  {r['timestamp']}  actual={r['load_kw']:.3f}  pred={r['load_pred']:.3f}  err={r['resid']:+.3f}  hour={r['hour']:>2}  dow={r['dow']}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(m["timestamp"], m["abs_err"], s=4, alpha=0.4, color="grey")
ax.scatter(worst["timestamp"], worst["abs_err"], s=80, color="red", marker="x", label="top-20 worst")
ax.set_ylabel("|residual| (kW)")
ax.set_title("Where the worst prediction errors happen in time")
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.tight_layout()
plt.savefig(OUT_PLOTS / "38_largest_errors.png", dpi=140, bbox_inches="tight")
plt.close()

# ── Plot 39: Error vs context ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0,0]
ax.scatter(m["lag_1"], m["resid"], s=4, alpha=0.3, color="#0066cc")
ax.axhline(0, color="red", ls="--")
ax.set_xlabel("lag_1 (kW)"); ax.set_ylabel("Residual (kW)")
ax.set_title("Residual vs lag_1")
ax.grid(alpha=0.3)

ax = axes[0,1]
ax.scatter(m["roll_4_std"], m["resid"], s=4, alpha=0.3, color="#cc3333")
ax.axhline(0, color="red", ls="--")
ax.set_xlabel("roll_4_std (recent volatility)"); ax.set_ylabel("Residual")
ax.set_title("Residual vs recent volatility")
ax.grid(alpha=0.3)

ax = axes[1,0]
ax.scatter(m["temperature_2m"], m["resid"], s=4, alpha=0.3, color="#9933cc")
ax.axhline(0, color="red", ls="--")
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Residual")
ax.set_title("Residual vs temperature")
ax.grid(alpha=0.3)

ax = axes[1,1]
ax.scatter(m["shortwave_radiation"], m["resid"], s=4, alpha=0.3, color="#cc6600")
ax.axhline(0, color="red", ls="--")
ax.set_xlabel("Shortwave radiation (W/m²)"); ax.set_ylabel("Residual")
ax.set_title("Residual vs solar radiation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOTS / "39_error_by_context.png", dpi=140, bbox_inches="tight")
plt.close()

# ── Plot 40: Under-prediction zones ──────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
m_sorted = m.sort_values("load_kw")
windows = np.array_split(m_sorted, 30)
mid_loads = [w["load_kw"].mean() for w in windows]
under_rates = [(w["resid"] > 0).mean() * 100 for w in windows]
mean_resid = [w["resid"].mean() for w in windows]
ax.plot(mid_loads, under_rates, "o-", color="green", label="% rows under-predicted")
ax.axhline(50, color="grey", ls="--", lw=1, label="50% (no bias)")
ax.set_xlabel("Mean actual load in bin (kW)"); ax.set_ylabel("% rows under-predicted")
ax.set_ylim(0, 100)
ax2 = ax.twinx()
ax2.plot(mid_loads, mean_resid, "s--", color="red", label="mean residual")
ax2.axhline(0, color="grey", ls=":")
ax2.set_ylabel("Mean residual (kW)")
ax.set_title("UNDER-PREDICTION zones — where model systematically misses spikes")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOTS / "40_underprediction_zones.png", dpi=140, bbox_inches="tight")
plt.close()

# ── Written analysis report ──────────────────────────────
mean_l = m["load_kw"].mean(); std_l = m["load_kw"].std()
worst_hour = hour_g["rmse"].idxmax()
best_hour  = hour_g["rmse"].idxmin()
worst_dow_name = dow_names[int(dow_g["nrmse"].idxmax())]
load_high = m["load_kw"] >= 1.5
high_under = (m.loc[load_high, "resid"] > 0).mean()
med_resid_high = m.loc[load_high, "resid"].median()

report = f"""
================================================================================
DIAGNOSTIC SUMMARY — best forecast (online_retraining, NRMSE 60.65%)
================================================================================

OVERALL
  N rows                      : {len(m)}
  Mean actual load            : {mean_l:.4f} kW
  Std actual load             : {std_l:.4f} kW
  CV (std/mean)               : {std_l/mean_l:.3f}    [>1 means high noise]
  Combined NRMSE              : {nrmse(m['load_kw'], m['load_pred']):.2f}%
  Combined RMSE               : {rmse(m['load_kw'], m['load_pred']):.4f} kW
  Combined MAE                : {m['abs_err'].mean():.4f} kW
  R²                          : {1 - np.sum(m['resid']**2)/np.sum((m['load_kw']-mean_l)**2):.3f}

WHERE THE MODEL FAILS
  Worst hour-of-day           : hour {worst_hour:>2} (RMSE {hour_g.loc[worst_hour, 'rmse']:.4f} kW)
  Best hour-of-day            : hour {best_hour:>2} (RMSE {hour_g.loc[best_hour, 'rmse']:.4f} kW)
  Worst day-of-week           : {worst_dow_name} (NRMSE {dow_g['nrmse'].max():.2f}%)
  Largest error overall       : {m['abs_err'].max():.4f} kW at {m.loc[m['abs_err'].idxmax(), 'timestamp']}

HETEROSCEDASTICITY (RMSE vs load level)
"""
for bin_label, row in load_g.iterrows():
    report += f"  Load {str(bin_label):>10s}            : RMSE {row['rmse']:.4f} kW   (n={row['n']})\n"

report += f"""
SPIKE UNDER-PREDICTION
  High-load rows (>=1.5 kW)   : {load_high.sum()}  ({load_high.mean()*100:.1f}% of test)
  Under-prediction rate (high): {high_under*100:.1f}%   [50% would mean unbiased]
  Median residual on high     : {med_resid_high:+.4f} kW   [positive = systematic under-prediction]

ROOT-CAUSE ANALYSIS
  1. CV = {std_l/mean_l:.2f} — std exceeds mean. 15-min residential load is heavy-tailed.
  2. lag-1 ACF ≈ 0.80; the IRREDUCIBLE 1-step floor from lag-1 alone is ~69% NRMSE.
     Our 60.65% beats that by capturing extra structure from rolling stats / weather.
  3. Worst regime: high-load periods (>=1.5 kW). RMSE jumps {load_g.loc['3+', 'rmse']/load_g.loc['0-0.5', 'rmse']:.1f}x
     vs low-load. These are appliance-switching events that:
       - happen unpredictably (not in training)
       - violate the smooth conditional mean the model targets
  4. Hour {worst_hour:>2} is hardest — this is when many simultaneous appliance events occur.
  5. The remaining variance is appliance-switching noise. To reduce further you need:
       - sub-meter data per appliance (impossible with current dataset)
       - aggregate to >5 households (impossible with this site)
       - 1-hour resolution metric (legal but changes the scoring)

WHAT WE CAN STILL TRY
  - Asymmetric loss (penalise under-prediction more) — tested, marginal
  - Ensemble with different seeds — already done (60.80% best)
  - Spike-detection classifier + amplitude regression — tested, hurt val-test gap
  - Cooling/heating threshold features (cooling_load_244) — being tested in v6
  - Italian-pulse features (lunch_peak, tou_shift) — being tested in v6
  - Wavelet decomposition on recent load — not yet tested (computationally heavy)
  - Probabilistic forecasting (predict quantiles) — would change what optimizer uses

================================================================================
"""

OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.write_text(report)
print(report)
print(f"\nFull report saved -> {OUT_REPORT}")
print(f"Plots saved        -> {OUT_PLOTS}/")
