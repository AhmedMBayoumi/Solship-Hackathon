"""
Comprehensive forecast-quality diagnostics.
Generates 8 plots in outputs/plots/forecasting/:
  20_pred_vs_actual_scatter.png
  21_residual_histogram.png
  22_error_breakdown_by_hour.png
  23_error_breakdown_by_dow.png
  24_error_vs_load_level.png
  25_error_vs_pv.png
  26_timeseries_predictions_week.png
  27_worst_best_predictions.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parents[1]
OUT  = ROOT / "outputs/plots/forecasting"
OUT.mkdir(parents=True, exist_ok=True)

# Load actuals + predictions from multiple models
df_test = pd.read_parquet(ROOT / "data/features/features_v2_test.parquet")
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])

PREDS = {
    "LightGBM v1":     "lgbm_test_preds.csv",
    "XGBoost v1":      "xgb_test_preds.csv",
    "Bagging (12×LGBM)": "bagging_test_preds.csv",
    "Stacked v2":      "stacked_v2_test_preds.csv",
}
preds = {}
for name, f in PREDS.items():
    p = ROOT / "outputs/forecasts" / f
    if not p.exists():
        print(f"  skip {name}: {p.name} not found")
        continue
    d = pd.read_csv(p, parse_dates=["timestamp"])
    preds[name] = df_test[["timestamp", "load_kw"]].merge(d, on="timestamp", how="left")

# Pick the BEST model for diagnostics
BEST = "Bagging (12×LGBM)"
m = preds[BEST]
y     = m["load_kw"].values
yhat  = m["load_pred"].values
mask  = ~np.isnan(yhat)
y, yhat = y[mask], yhat[mask]
sub  = m[mask].reset_index(drop=True)
sub["pred"]  = yhat
sub["resid"] = sub["load_kw"] - sub["pred"]
sub["abs_err"] = sub["resid"].abs()
sub["hour"]    = sub["timestamp"].dt.hour
sub["dow"]     = sub["timestamp"].dt.dayofweek
sub["month"]   = sub["timestamp"].dt.month
sub["pv_kw"]   = df_test.set_index("timestamp")["pv_kw"].reindex(sub["timestamp"]).values

def nrmse(y, yp): return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)
def rmse(y, yp):  return float(np.sqrt(np.mean((y - yp) ** 2)))

print(f"=== Diagnosing {BEST} ===")
print(f"  N: {len(sub)}  RMSE: {rmse(y, yhat):.4f}  NRMSE: {nrmse(y, yhat):.2f}%  R2: {1 - np.sum((y-yhat)**2)/np.sum((y-y.mean())**2):.3f}\n")


# ── 1. Predicted vs actual scatter ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(yhat, y, alpha=0.15, s=8, color="#0066cc")
mx = max(y.max(), yhat.max()) * 1.05
ax.plot([0, mx], [0, mx], "r--", lw=1.5, label="y = ŷ (perfect)")
ax.set_xlabel("Predicted load (kW)")
ax.set_ylabel("Actual load (kW)")
ax.set_title(f"Predicted vs actual — {BEST}\nNRMSE = {nrmse(y, yhat):.2f}%, RMSE = {rmse(y, yhat):.4f} kW")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "20_pred_vs_actual_scatter.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 2. Residual histogram ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(sub["resid"], bins=80, color="#0066cc", alpha=0.75, edgecolor="black")
axes[0].axvline(0, color="red", ls="--", lw=1.5)
axes[0].axvline(sub["resid"].mean(), color="green", ls="-", lw=1.5,
                label=f"mean = {sub['resid'].mean():+.4f}")
axes[0].set_xlabel("Residual (actual − predicted) [kW]")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Residual distribution — {BEST}")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(sub["abs_err"], bins=80, color="#cc6600", alpha=0.75, edgecolor="black")
axes[1].axvline(sub["abs_err"].mean(), color="green", ls="-", lw=1.5,
                label=f"MAE = {sub['abs_err'].mean():.4f} kW")
axes[1].axvline(sub["abs_err"].median(), color="purple", ls=":", lw=1.5,
                label=f"median = {sub['abs_err'].median():.4f}")
axes[1].set_xlabel("Absolute error |actual − predicted| [kW]")
axes[1].set_ylabel("Count")
axes[1].set_title("Absolute error distribution")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "21_residual_histogram.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 3. Error breakdown by hour ──────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 5))
hour_grp = sub.groupby("hour").agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    mae =("abs_err", "mean"),
    bias=("resid", "mean"),
    cnt =("load_kw", "count"),
    actual_mean=("load_kw", "mean"),
)
hour_grp["nrmse"] = hour_grp["rmse"] / hour_grp["actual_mean"] * 100
ax1.bar(hour_grp.index, hour_grp["rmse"], color="#0066cc", alpha=0.7, label="RMSE (kW)")
ax1.set_xlabel("Hour of day")
ax1.set_ylabel("RMSE (kW)", color="#0066cc")
ax1.tick_params(axis="y", labelcolor="#0066cc")
ax1.set_xticks(range(24))
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(hour_grp.index, hour_grp["bias"], "o-", color="red", lw=2, label="Bias (mean residual)")
ax2.axhline(0, color="black", lw=0.5, ls="--")
ax2.set_ylabel("Bias (kW)", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax1.set_title(f"Error by hour-of-day — {BEST}")
fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
plt.tight_layout()
plt.savefig(OUT / "22_error_breakdown_by_hour.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 4. Error breakdown by day-of-week ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
DOW_NAME = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_grp = sub.groupby("dow").agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    mae =("abs_err", "mean"),
    bias=("resid", "mean"),
    actual_mean=("load_kw", "mean"),
)
dow_grp["nrmse"] = dow_grp["rmse"] / dow_grp["actual_mean"] * 100
axes[0].bar(range(7), dow_grp["nrmse"], color="#9933cc", alpha=0.75)
axes[0].set_xticks(range(7)); axes[0].set_xticklabels(DOW_NAME)
axes[0].set_ylabel("NRMSE (%)")
axes[0].set_title("NRMSE by day-of-week")
axes[0].grid(alpha=0.3)

axes[1].bar(range(7), dow_grp["bias"], color="orange", alpha=0.75)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_xticks(range(7)); axes[1].set_xticklabels(DOW_NAME)
axes[1].set_ylabel("Bias (kW)")
axes[1].set_title("Mean residual (bias) by day-of-week")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "23_error_breakdown_by_dow.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 5. Error vs load level (heteroscedasticity) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Bin actual load
sub["load_bin"] = pd.cut(sub["load_kw"], bins=[-0.01, 0.5, 1.0, 1.5, 2.0, 3.0, 100], labels=["0-0.5","0.5-1","1-1.5","1.5-2","2-3","3+"])
err_bin = sub.groupby("load_bin", observed=True).agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    cnt=("load_kw","count"),
    bias=("resid","mean"),
)
axes[0].bar(range(len(err_bin)), err_bin["rmse"], color="#cc3333", alpha=0.75)
axes[0].set_xticks(range(len(err_bin))); axes[0].set_xticklabels(err_bin.index, rotation=0)
axes[0].set_xlabel("Actual load bin (kW)")
axes[0].set_ylabel("RMSE (kW)")
axes[0].set_title("Error grows with load level (heteroscedasticity)")
for i, c in enumerate(err_bin["cnt"]):
    axes[0].text(i, err_bin["rmse"].iloc[i] + 0.02, f"n={c}", ha="center", fontsize=8)
axes[0].grid(alpha=0.3)

# scatter resid vs load
axes[1].scatter(sub["load_kw"], sub["resid"], alpha=0.1, s=6, color="#0066cc")
axes[1].axhline(0, color="red", ls="--", lw=1.5)
axes[1].set_xlabel("Actual load (kW)")
axes[1].set_ylabel("Residual (kW)")
axes[1].set_title("Residuals vs actual load")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "24_error_vs_load_level.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 6. Error vs PV level ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sub["pv_bin"] = pd.cut(sub["pv_kw"], bins=[-0.01, 0.1, 1, 3, 5, 100], labels=["0","0.1-1","1-3","3-5","5+"])
pv_bin = sub.groupby("pv_bin", observed=True).agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    cnt=("load_kw","count"),
    bias=("resid","mean"),
    actual_mean=("load_kw","mean"),
)
pv_bin["nrmse"] = pv_bin["rmse"] / pv_bin["actual_mean"] * 100
x = range(len(pv_bin))
w = 0.35
ax.bar([i - w/2 for i in x], pv_bin["rmse"], w, color="#cc3333", label="RMSE (kW)")
ax2 = ax.twinx()
ax2.bar([i + w/2 for i in x], pv_bin["nrmse"], w, color="#0066cc", alpha=0.7, label="NRMSE (%)")
ax.set_xticks(x); ax.set_xticklabels(pv_bin.index)
ax.set_xlabel("PV bin (kW)")
ax.set_ylabel("RMSE (kW)", color="#cc3333")
ax2.set_ylabel("NRMSE (%)", color="#0066cc")
ax.set_title("Forecast error vs PV generation level")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "25_error_vs_pv.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 7. Timeseries: 1 week of pred vs actual ────────────────────
# Pick April week 2 (Apr 8-14, 2025)
WEEK_START = pd.Timestamp("2025-04-08")
WEEK_END   = pd.Timestamp("2025-04-14 23:59:59")
wk = sub[(sub["timestamp"] >= WEEK_START) & (sub["timestamp"] <= WEEK_END)].copy()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(wk["timestamp"], wk["load_kw"], color="#0066cc", lw=1.0, label="actual")
axes[0].plot(wk["timestamp"], wk["pred"],    color="#cc3333", lw=1.0, ls="--", label="predicted", alpha=0.8)
axes[0].fill_between(wk["timestamp"], wk["load_kw"], wk["pred"],
                     where=(wk["resid"] > 0), alpha=0.2, color="green",
                     label="under-prediction")
axes[0].fill_between(wk["timestamp"], wk["load_kw"], wk["pred"],
                     where=(wk["resid"] < 0), alpha=0.2, color="red",
                     label="over-prediction")
axes[0].set_ylabel("Load (kW)")
axes[0].set_title(f"April Week 2 2025 — {BEST}, week NRMSE = {nrmse(wk['load_kw'], wk['pred']):.2f}%")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="upper right")

axes[1].plot(wk["timestamp"], wk["resid"], color="#9933cc", lw=1.0)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].fill_between(wk["timestamp"], 0, wk["resid"],
                     where=(wk["resid"] > 0), alpha=0.3, color="green")
axes[1].fill_between(wk["timestamp"], 0, wk["resid"],
                     where=(wk["resid"] < 0), alpha=0.3, color="red")
axes[1].set_ylabel("Residual (kW)")
axes[1].set_xlabel("Date")
axes[1].grid(alpha=0.3)
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %m-%d"))
plt.tight_layout()
plt.savefig(OUT / "26_timeseries_predictions_week.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 8. Worst & best windows ────────────────────────────────────
sub["day"] = sub["timestamp"].dt.normalize()
day_err = sub.groupby("day").agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    mae=("abs_err", "mean"),
    actual_sum=("load_kw","sum"),
    pred_sum  =("pred","sum"),
).sort_values("rmse")
worst = day_err.tail(1).index[0]
best  = day_err.head(1).index[0]

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
for ax, day, kind in [(axes[0], best, "BEST"), (axes[1], worst, "WORST")]:
    d = sub[sub["day"] == day]
    ax.plot(d["timestamp"], d["load_kw"], color="#0066cc", lw=1.5, label="actual")
    ax.plot(d["timestamp"], d["pred"], color="#cc3333", lw=1.5, ls="--", label="predicted")
    e = day_err.loc[day]
    ax.set_title(f"{kind} day: {day.date()}  RMSE={e['rmse']:.4f}  MAE={e['mae']:.4f}")
    ax.set_ylabel("Load (kW)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[1].set_xlabel("Time")
plt.tight_layout()
plt.savefig(OUT / "27_worst_best_predictions.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 9. Multi-model comparison ──────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
nrmses = {}
maes = {}
for name, m in preds.items():
    y_  = m["load_kw"].values
    yh_ = m["load_pred"].values
    mk  = ~np.isnan(yh_)
    nrmses[name] = nrmse(y_[mk], yh_[mk])
    maes[name]   = float(np.mean(np.abs(y_[mk] - yh_[mk])))
names = list(nrmses)
x = np.arange(len(names))
ax.bar(x, [nrmses[n] for n in names], color="#0066cc", alpha=0.75, label="NRMSE (%)")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha="right")
ax.set_ylabel("Test NRMSE (%)", color="#0066cc")
ax.axhline(72.30, color="red", ls=":", lw=1, label="lag-1 baseline (72%)")
ax.set_title("Test NRMSE comparison across model variants — Apr+Sep 2025")
ax.grid(alpha=0.3)
for i, v in enumerate([nrmses[n] for n in names]):
    ax.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontsize=10)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "28_model_comparison.png", dpi=140, bbox_inches="tight")
plt.close()

print(f"Plots saved to {OUT}")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}")
