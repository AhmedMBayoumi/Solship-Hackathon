"""
Build all submission files per the hackathon supervisor's spec:
  - April forecast CSV (timestamp, actual, predicted, error)
  - September forecast CSV (timestamp, actual, predicted, error)
  - Master metrics CSV (baselines + all errors)
  - Bills + savings CSV
  - Forecast comparison plots (per month)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.eval.compute_bill import compute_bill, baseline_a_bill, baseline_b_bill

ROOT = Path(__file__).parents[1]
SUB  = ROOT / "submission"
SUB_DATA  = SUB / "data"
SUB_PLOTS = SUB / "plots"
SUB_DATA.mkdir(parents=True, exist_ok=True)
SUB_PLOTS.mkdir(parents=True, exist_ok=True)

# ── Load best forecast (walkforward bagging) and actuals ──
df = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df[df["timestamp"].dt.year == 2025].copy()
preds = pd.read_csv(ROOT / "outputs/forecasts/bagging_walkforward_test_preds.csv", parse_dates=["timestamp"])

merged = df_2025[df_2025["timestamp"].dt.month.isin([4, 9])][["timestamp","load_kw","pv_kw","buy_price","sell_price"]].merge(
    preds, on="timestamp", how="left"
)
merged["abs_error"] = (merged["load_kw"] - merged["load_pred"]).abs()
merged["sq_error"]  = (merged["load_kw"] - merged["load_pred"]) ** 2

# ── 1. April forecast CSV ─────────────────────────────────
apr = merged[merged["timestamp"].dt.month == 4].copy()
apr_out = apr[["timestamp","load_kw","load_pred","abs_error"]].rename(
    columns={"load_kw":"load_actual_kW","load_pred":"load_forecast_kW","abs_error":"abs_error_kW"})
apr_out.to_csv(SUB_DATA / "forecast_april_2025.csv", index=False)
print(f"April forecast CSV: {len(apr_out)} rows -> forecast_april_2025.csv")

# ── 2. September forecast CSV ─────────────────────────────
sep = merged[merged["timestamp"].dt.month == 9].copy()
sep_out = sep[["timestamp","load_kw","load_pred","abs_error"]].rename(
    columns={"load_kw":"load_actual_kW","load_pred":"load_forecast_kW","abs_error":"abs_error_kW"})
sep_out.to_csv(SUB_DATA / "forecast_september_2025.csv", index=False)
print(f"September forecast CSV: {len(sep_out)} rows -> forecast_september_2025.csv")

# ── 3. Per-month metrics ──────────────────────────────────
def metrics(y, yp):
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    mae  = float(np.mean(np.abs(y - yp)))
    nrm  = rmse / np.mean(y) * 100
    r2   = 1 - np.sum((y - yp) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return {"RMSE_kW": round(rmse,4), "MAE_kW": round(mae,4),
            "NRMSE_%": round(nrm,2), "R2": round(r2,4),
            "n": len(y), "mean_actual_kW": round(float(np.mean(y)),4)}

apr_m = metrics(apr["load_kw"].values, apr["load_pred"].values)
sep_m = metrics(sep["load_kw"].values, sep["load_pred"].values)
all_m = metrics(merged["load_kw"].values, merged["load_pred"].values)

# ── 4. Master metrics CSV ─────────────────────────────────
def per_month_bill(bf, df_):
    out = {}
    for m, n in [(4,"April"),(9,"September")]:
        sub = df_[df_["timestamp"].dt.month == m]
        out[n] = bf(sub)
    return out
A_per = per_month_bill(baseline_a_bill, df_2025)
B_per = per_month_bill(baseline_b_bill, df_2025)

# Our MPC: walkforward H=96
mpc_path = ROOT / "outputs/mpc_walkforward_H96.parquet"
mpc_df = pd.read_parquet(mpc_path)
mpc_per = {}
for m, n in [(4,"April"),(9,"September")]:
    sub = mpc_df[mpc_df["timestamp"].dt.month == m]
    mpc_per[n] = compute_bill(sub, sub["p_grid_kw"])

oracle_per = {
    "April":     {"net_bill": -20.13, "import_cost": 13.43, "export_revenue": 33.56},
    "September": {"net_bill":  -0.02, "import_cost": 18.66, "export_revenue": 18.68},
}

bill_rows = []
for label, per in [
    ("Baseline A (existing controller)", A_per),
    ("Baseline B (no battery)",           B_per),
    ("Our MPC (walkforward bagging, H=96)", mpc_per),
    ("Oracle (perfect foresight, H=96)",  oracle_per),
]:
    a = per["April"]["net_bill"]; s = per["September"]["net_bill"]
    a_imp = per["April"].get("import_cost", 0); a_exp = per["April"].get("export_revenue", 0)
    s_imp = per["September"].get("import_cost", 0); s_exp = per["September"].get("export_revenue", 0)
    bill_rows.append({
        "controller": label,
        "April_import_EUR":  round(a_imp, 2),
        "April_export_EUR":  round(a_exp, 2),
        "April_net_EUR":     round(a, 2),
        "Sept_import_EUR":   round(s_imp, 2),
        "Sept_export_EUR":   round(s_exp, 2),
        "Sept_net_EUR":      round(s, 2),
        "Total_EUR":         round(a + s, 2),
    })
bill_df = pd.DataFrame(bill_rows)
A_tot = bill_df.iloc[0]["Total_EUR"]
bill_df["Savings_vs_A_EUR"] = (A_tot - bill_df["Total_EUR"]).round(2)
bill_df["Savings_vs_A_pct"] = (bill_df["Savings_vs_A_EUR"] / abs(A_tot) * 100).round(1)
print(bill_df.to_string(index=False))
bill_df.to_csv(SUB_DATA / "bills_and_savings.csv", index=False)

# Master metrics
with open(SUB_DATA / "master_metrics.csv", "w", encoding="utf-8") as f:
    f.write("# Solship Hackathon 2026 -- Master metrics\n")
    f.write("# Test window: April 2025 + September 2025\n")
    f.write("# Best model: walkforward bagging (12x LightGBM, trained with 2024+2025 up to test month)\n")
    f.write("# MPC horizon: H=96 (1 day, per supervisor's recommendation)\n\n")
    f.write("== FORECAST QUALITY ==\n")
    f.write("month,RMSE_kW,MAE_kW,NRMSE_%,R2,n,mean_actual_kW\n")
    f.write(f"April,{apr_m['RMSE_kW']},{apr_m['MAE_kW']},{apr_m['NRMSE_%']},{apr_m['R2']},{apr_m['n']},{apr_m['mean_actual_kW']}\n")
    f.write(f"September,{sep_m['RMSE_kW']},{sep_m['MAE_kW']},{sep_m['NRMSE_%']},{sep_m['R2']},{sep_m['n']},{sep_m['mean_actual_kW']}\n")
    f.write(f"Combined,{all_m['RMSE_kW']},{all_m['MAE_kW']},{all_m['NRMSE_%']},{all_m['R2']},{all_m['n']},{all_m['mean_actual_kW']}\n")
    f.write("\n== CONTROLLER BILLS (EUR) ==\n")
    bill_df.to_csv(f, index=False)

print("\n=== METRICS ===")
print(f"April     : RMSE={apr_m['RMSE_kW']:.4f}  MAE={apr_m['MAE_kW']:.4f}  NRMSE={apr_m['NRMSE_%']:.2f}%  R2={apr_m['R2']:.3f}")
print(f"September : RMSE={sep_m['RMSE_kW']:.4f}  MAE={sep_m['MAE_kW']:.4f}  NRMSE={sep_m['NRMSE_%']:.2f}%  R2={sep_m['R2']:.3f}")
print(f"Combined  : RMSE={all_m['RMSE_kW']:.4f}  MAE={all_m['MAE_kW']:.4f}  NRMSE={all_m['NRMSE_%']:.2f}%  R2={all_m['R2']:.3f}")

# ── 5. Forecast comparison plots ──────────────────────────
for month, name, df_m in [(4, "April", apr), (9, "September", sep)]:
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(df_m["timestamp"], df_m["load_kw"], color="#0066cc", lw=0.8, label="Actual")
    axes[0].plot(df_m["timestamp"], df_m["load_pred"], color="#cc3333", lw=0.8, ls="--", label="Forecast (bagging walkforward)")
    axes[0].set_ylabel("Load (kW)")
    axes[0].set_title(f"{name} 2025  --  Load forecast vs actual\n"
                      f"NRMSE={metrics(df_m['load_kw'].values, df_m['load_pred'].values)['NRMSE_%']:.2f}%  "
                      f"RMSE={metrics(df_m['load_kw'].values, df_m['load_pred'].values)['RMSE_kW']:.4f} kW  "
                      f"MAE={metrics(df_m['load_kw'].values, df_m['load_pred'].values)['MAE_kW']:.4f} kW")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    err = df_m["load_kw"] - df_m["load_pred"]
    axes[1].plot(df_m["timestamp"], err, color="#9933cc", lw=0.8)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].fill_between(df_m["timestamp"], 0, err, alpha=0.2,
                          where=(err > 0), color="green", label="under-prediction")
    axes[1].fill_between(df_m["timestamp"], 0, err, alpha=0.2,
                          where=(err < 0), color="red", label="over-prediction")
    axes[1].set_ylabel("Residual (kW)")
    axes[1].set_xlabel("Date")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    out = SUB_PLOTS / f"forecast_{name.lower()}_2025.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")

# ── 6. Bill comparison bar chart ──────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
labels = bill_df["controller"].tolist()
totals = bill_df["Total_EUR"].tolist()
colors = ["#888","#cc6666","#33aa33","#0066cc"]
bars = ax.bar(range(len(labels)), totals, color=colors)
for i, t in enumerate(totals):
    ax.text(i, t + (0.5 if t >= 0 else -2), f"EUR {t:+.2f}", ha="center", fontweight="bold")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)
ax.set_ylabel("Total bill April + September 2025 (EUR)")
ax.set_title("Net electricity bill (negative = profit)\nHackathon submission: walkforward-bagging MPC vs baselines & oracle")
ax.axhline(0, color="black", lw=0.5)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
out = SUB_PLOTS / "bills_comparison.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved -> {out}")

# Copy already-existing key plots into submission
import shutil
KEY_PLOTS = [
    ("outputs/plots/eda/06_load_heatmap_hour_dow.png", "eda_load_heatmap.png"),
    ("outputs/plots/eda/13_price_analysis.png", "eda_price_analysis.png"),
    ("outputs/plots/eda/16_autocorrelation.png", "eda_autocorrelation.png"),
    ("outputs/plots/forecasting/20_pred_vs_actual_scatter.png", "forecast_pred_vs_actual.png"),
    ("outputs/plots/forecasting/22_error_breakdown_by_hour.png", "forecast_error_by_hour.png"),
    ("outputs/plots/forecasting/29_feature_importance_top30.png", "forecast_feature_importance.png"),
    ("outputs/plots/forecasting/34_spike_event_analysis.png", "forecast_spike_analysis.png"),
    ("outputs/plots/results/19_horizon_sensitivity.png", "results_horizon_sensitivity.png"),
    ("outputs/plots/dispatch/march_week3_dispatch.png", "dispatch_march_week3.png"),
]
for src, dst in KEY_PLOTS:
    p = ROOT / src
    if p.exists():
        shutil.copy(p, SUB_PLOTS / dst)
        print(f"Copied -> {dst}")

print(f"\nAll submission files written to {SUB}")
