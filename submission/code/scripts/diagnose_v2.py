"""
Round 2 of forecast diagnostics. Generates 6 more plots:
  29_feature_importance_top30.png
  30_residual_autocorrelation.png
  31_pred_vs_actual_by_hour.png
  32_per_month_breakdown.png
  33_error_vs_recent_change.png
  34_spike_event_analysis.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
OUT  = ROOT / "outputs/plots/forecasting"

# Load test set + bagging predictions (best so far)
df_test = pd.read_parquet(ROOT / "data/features/features_v2_test.parquet")
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
preds = pd.read_csv(ROOT / "outputs/forecasts/bagging_test_preds.csv", parse_dates=["timestamp"])
m = df_test.merge(preds, on="timestamp", how="left")
m = m.dropna(subset=["load_pred"]).reset_index(drop=True)
m["resid"] = m["load_kw"] - m["load_pred"]
m["abs_err"] = m["resid"].abs()
m["hour"] = m["timestamp"].dt.hour
m["dow"]  = m["timestamp"].dt.dayofweek
m["month"]= m["timestamp"].dt.month

def nrmse(y, yp): return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)

# ── 29. Feature importance from a fresh LGBM (heavy reg) ─────
print("Training LGBM to get feature importance...")
DROP_BASE  = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
              "qow","hod","net_load","sell_price","pv_today_total"}
DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
train_df = pd.read_parquet(ROOT / "data/features/features_v2_train.parquet")
val_df   = pd.read_parquet(ROOT / "data/features/features_v2_val.parquet")
feats = [c for c in train_df.columns if c not in (DROP_BASE | DROP_LEAKY)]
X_tv = np.vstack([train_df[feats].values, val_df[feats].values])
y_tv = np.concatenate([train_df["load_kw"].values, val_df["load_kw"].values])
mdl = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
    min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    objective="huber", alpha=0.9, verbose=-1, n_jobs=-1, random_state=42,
)
mdl.fit(X_tv, y_tv)

imp_df = pd.DataFrame({"feature": feats, "importance": mdl.feature_importances_}).sort_values("importance", ascending=False)
top30 = imp_df.head(30)

fig, ax = plt.subplots(figsize=(10, 9))
y_pos = np.arange(len(top30))
ax.barh(y_pos, top30["importance"].values[::-1], color="#0066cc", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top30["feature"].values[::-1], fontsize=9)
ax.set_xlabel("Importance (split count)")
ax.set_title("Top-30 LightGBM features  (heavy-reg + huber)")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(OUT / "29_feature_importance_top30.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 30. Residual autocorrelation (are residuals random?) ─────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
r = m["resid"].values
lags = list(range(1, 50)) + [96, 192, 288, 384, 672]
acf = [float(np.corrcoef(r[lag:], r[:-lag])[0,1]) for lag in lags]
axes[0].bar(lags, acf, color="#0066cc", alpha=0.8)
axes[0].axhline(0, color="black", lw=0.5)
axes[0].axhline(0.1, color="red", lw=0.5, ls="--", label="±0.1")
axes[0].axhline(-0.1, color="red", lw=0.5, ls="--")
axes[0].set_xlabel("Lag (15-min steps)")
axes[0].set_ylabel("ACF of residuals")
axes[0].set_title("Residual autocorrelation\n(non-zero = there's structure we're missing)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xscale("symlog")

# Spectral analysis of residuals
freqs = np.fft.rfftfreq(len(r), d=15.0)  # cycles per minute
spec = np.abs(np.fft.rfft(r - r.mean()))
period_min = np.where(freqs > 0, 1.0/freqs, np.inf)
period_h   = period_min / 60.0
mask_show = (period_h > 0.5) & (period_h < 200)
axes[1].plot(period_h[mask_show], spec[mask_show], color="#cc6600")
for p, lab in [(24, "24h"), (12, "12h"), (8, "8h"), (168, "1wk")]:
    axes[1].axvline(p, color="green", lw=1, ls=":", alpha=0.7)
    axes[1].text(p, spec[mask_show].max() * 0.95, lab, color="green", ha="center", fontsize=9)
axes[1].set_xscale("log")
axes[1].set_xlabel("Period (hours)")
axes[1].set_ylabel("Magnitude")
axes[1].set_title("Residual spectrum\n(peaks = unexplained periodic patterns)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "30_residual_autocorrelation.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 31. Predicted vs actual colored by hour ────────────────
fig, ax = plt.subplots(figsize=(9, 8))
sc = ax.scatter(m["load_pred"], m["load_kw"], c=m["hour"], s=8, alpha=0.4, cmap="twilight")
mx = max(m["load_kw"].max(), m["load_pred"].max()) * 1.05
ax.plot([0, mx], [0, mx], "r--", lw=1.5, label="y = ŷ")
ax.set_xlabel("Predicted (kW)")
ax.set_ylabel("Actual (kW)")
ax.set_title("Pred vs actual, colored by hour-of-day")
plt.colorbar(sc, ax=ax, label="Hour of day")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "31_pred_vs_actual_by_hour.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 32. Per-month and per-week breakdown ──────────────────
m["week"] = m["timestamp"].dt.isocalendar().week.astype(int)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
month_grp = m.groupby("month").agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    actual_mean=("load_kw","mean"),
    n=("load_kw","count"),
)
month_grp["nrmse"] = month_grp["rmse"] / month_grp["actual_mean"] * 100
axes[0].bar(month_grp.index.astype(str), month_grp["nrmse"], color="#9933cc", alpha=0.75)
axes[0].set_xlabel("Month (2025)")
axes[0].set_ylabel("NRMSE (%)")
axes[0].set_title("NRMSE by month")
axes[0].grid(alpha=0.3)
for i, (mo, v) in enumerate(month_grp["nrmse"].items()):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")

week_grp = m.groupby(["month","week"]).agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    actual_mean=("load_kw","mean"),
    n=("load_kw","count"),
)
week_grp["nrmse"] = week_grp["rmse"] / week_grp["actual_mean"] * 100
labels = [f"M{mo}-W{w}" for (mo,w) in week_grp.index]
axes[1].bar(range(len(week_grp)), week_grp["nrmse"], color="#cc6600", alpha=0.75)
axes[1].set_xticks(range(len(week_grp)))
axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("NRMSE (%)")
axes[1].set_title("NRMSE by ISO week")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "32_per_month_breakdown.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 33. Error vs recent-change features ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
m["d_lag1"] = df_test["d_lag1"].values
m["lag_1"]  = df_test["lag_1"].values
# Bin by recent change (volatility)
m["dvol"] = np.abs(m["d_lag1"])
m["dvol_bin"] = pd.cut(m["dvol"], bins=[-0.01, 0.1, 0.3, 0.6, 1.0, 100], labels=["0-0.1","0.1-0.3","0.3-0.6","0.6-1","1+"])
vol_err = m.groupby("dvol_bin", observed=True).agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    n=("load_kw","count"),
)
axes[0].bar(range(len(vol_err)), vol_err["rmse"], color="#cc3333", alpha=0.75)
axes[0].set_xticks(range(len(vol_err))); axes[0].set_xticklabels(vol_err.index)
axes[0].set_xlabel("|load[t-1] - load[t-2]| bin (kW) — recent volatility")
axes[0].set_ylabel("RMSE (kW)")
axes[0].set_title("Error grows with recent volatility")
for i, (n, v) in enumerate(zip(vol_err["n"], vol_err["rmse"])):
    axes[0].text(i, v + 0.02, f"n={n}", ha="center", fontsize=8)
axes[0].grid(alpha=0.3)

# Error vs lag_1 level
lag1_bin = pd.cut(m["lag_1"], bins=[-0.01,0.5,1,1.5,2,100], labels=["0-0.5","0.5-1","1-1.5","1.5-2","2+"])
m["lag1_bin"] = lag1_bin
lag1_err = m.groupby("lag1_bin", observed=True).agg(
    rmse=("resid", lambda x: np.sqrt((x**2).mean())),
    bias=("resid","mean"),
)
axes[1].bar(range(len(lag1_err)), lag1_err["bias"], color="orange", alpha=0.75)
axes[1].set_xticks(range(len(lag1_err))); axes[1].set_xticklabels(lag1_err.index)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_xlabel("lag_1 bin (kW)")
axes[1].set_ylabel("Bias (kW)")
axes[1].set_title("Bias by lag_1 level\n(positive = under-prediction; high lag → bigger spike to predict)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "33_error_vs_recent_change.png", dpi=140, bbox_inches="tight")
plt.close()

# ── 34. Spike-event analysis ───────────────────────────────
# Define a "spike" as load > 2 kW AND load > 1.5 * recent rolling mean
m["spike"] = (m["load_kw"] > 2.0).astype(int)
spike_count = m["spike"].sum()
nonspike_count = len(m) - spike_count
spike_err  = m[m["spike"]==1]["resid"]
nonspike_err = m[m["spike"]==0]["resid"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist([nonspike_err, spike_err], bins=50, label=["non-spike (load<2 kW)", "spike (load>=2 kW)"],
             color=["#0066cc","#cc3333"], alpha=0.65, stacked=False, density=True)
axes[0].axvline(0, color="black", ls="--", lw=1)
axes[0].set_xlabel("Residual (actual - predicted) [kW]")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Residual distribution: spike vs non-spike\n"
                  f"spike: n={spike_count} mean_resid={spike_err.mean():+.3f}  "
                  f"non-spike: n={nonspike_count} mean_resid={nonspike_err.mean():+.3f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

# How well does the model PREDICT spikes? Confusion-style
m["spike_pred"] = (m["load_pred"] > 2.0).astype(int)
TP = ((m["spike"]==1) & (m["spike_pred"]==1)).sum()
FN = ((m["spike"]==1) & (m["spike_pred"]==0)).sum()
FP = ((m["spike"]==0) & (m["spike_pred"]==1)).sum()
TN = ((m["spike"]==0) & (m["spike_pred"]==0)).sum()
recall    = TP / (TP + FN) if (TP+FN) > 0 else 0
precision = TP / (TP + FP) if (TP+FP) > 0 else 0

axes[1].bar(["TN", "FP", "FN", "TP"], [TN, FP, FN, TP],
            color=["#666","#ffaa44","#cc3333","#33cc66"])
axes[1].set_ylabel("Count")
axes[1].set_title(f"Spike detection (load > 2 kW threshold)\n"
                  f"Recall={recall:.1%}  Precision={precision:.1%}")
for i, v in enumerate([TN, FP, FN, TP]):
    axes[1].text(i, v + 50, str(v), ha="center")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "34_spike_event_analysis.png", dpi=140, bbox_inches="tight")
plt.close()

print(f"Plots saved to {OUT}")
print(f"\n=== Spike detection (load > 2 kW) ===")
print(f"  Total spikes in test : {spike_count} / {len(m)} ({spike_count/len(m):.1%})")
print(f"  Recall = {recall:.1%}  (we catch this fraction of spikes)")
print(f"  Precision = {precision:.1%}  (this fraction of our spike predictions are real)")
print(f"  Spike residual mean = {spike_err.mean():+.3f} kW (under-prediction)")
print(f"\n=== Top 10 features ===")
print(imp_df.head(10).to_string(index=False))
print(f"\n=== Per-month NRMSE ===")
print(month_grp[["rmse","actual_mean","nrmse","n"]].round(3))
