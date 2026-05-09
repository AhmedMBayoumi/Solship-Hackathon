"""Build the final submission CSV with all baselines + all error metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from src.eval.compute_bill import compute_bill, baseline_a_bill, baseline_b_bill

ROOT = Path(__file__).parents[1]
SUB  = ROOT / "submission"

# Load actuals
df = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df[df["timestamp"].dt.year == 2025].copy()

# ── Forecast metrics for all attempts ──────────────────────
df_test = pd.read_parquet(ROOT / "data/features/features_test.parquet")
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
y_test = df_test["load_kw"].values

forecast_files = {
    "LightGBM v1 (39 features)":         "lgbm_test_preds.csv",
    "XGBoost v1 (39 features)":          "xgb_test_preds.csv",
    "LightGBM v2 + heavy reg + huber":   "lgbm_logblend_test_preds.csv",
    "Bagging 12x LGBM (v2 features)":    "bagging_test_preds.csv",
    "Stacked ensemble (LGBM+XGB+CAT)":   "stacked_v2_test_preds.csv",
    "ML+DL Gated Fusion v1":             None,  # didn't save preds clean
    "ML+DL Gated Fusion v2 (spike)":     "fusion_v2_test_preds.csv",
    "ML+DL Gated Fusion v3 (specialists)": "fusion_v3_test_preds.csv",
    "Bagging walkforward (BEST)":        "bagging_walkforward_test_preds.csv",
}

def metrics(y, yp):
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    mae  = float(np.mean(np.abs(y - yp)))
    nrm  = rmse / np.mean(y) * 100 if np.mean(y) > 0 else 0.0
    r2   = 1 - np.sum((y - yp) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return rmse, mae, nrm, r2

rows = []
for name, fname in forecast_files.items():
    if fname is None:
        continue
    p = ROOT / "outputs/forecasts" / fname
    if not p.exists():
        continue
    df_p = pd.read_csv(p, parse_dates=["timestamp"])
    m = df_test[["timestamp", "load_kw"]].merge(df_p, on="timestamp", how="left")
    yt, yp = m["load_kw"].values, m["load_pred"].values
    mask = ~np.isnan(yp)
    rmse, mae, nrm, r2 = metrics(yt[mask], yp[mask])
    rows.append({
        "model": name,
        "n":     int(mask.sum()),
        "RMSE_kW":  round(rmse, 4),
        "MAE_kW":   round(mae, 4),
        "NRMSE_%":  round(nrm, 2),
        "R2":       round(r2, 4),
    })
forecast_df = pd.DataFrame(rows)
print("FORECAST QUALITY SUMMARY")
print(forecast_df.to_string(index=False))
forecast_df.to_csv(SUB / "data/forecast_metrics.csv", index=False)

# ── Bills for all controllers ──────────────────────────────
def per_month_bill(bf, df_):
    out = {}
    for m, n in [(4, "April"), (9, "September")]:
        sub = df_[df_["timestamp"].dt.month == m]
        out[n] = bf(sub)
    return out

A_per = per_month_bill(baseline_a_bill, df_2025)
B_per = per_month_bill(baseline_b_bill, df_2025)

def bill_from_dispatch(parquet_path):
    if not parquet_path.exists():
        return None
    res = pd.read_parquet(parquet_path)
    out = {}
    for m, n in [(4, "April"), (9, "September")]:
        sub = res[res["timestamp"].dt.month == m]
        out[n] = compute_bill(sub, sub["p_grid_kw"])
    return out

ORACLE_VAL = {
    "April":     {"net_bill": -20.13, "import_cost": 13.43, "export_revenue": 33.56},
    "September": {"net_bill":  -0.02, "import_cost": 18.66, "export_revenue": 18.68},
}

controllers = []
controllers.append(("Baseline A (existing controller)", A_per))
controllers.append(("Baseline B (no battery)",          B_per))
controllers.append(("Oracle (perfect foresight, H=96)", ORACLE_VAL))

# Our MPC results (read from saved dispatches)
for H in [96, 192, 672]:
    p = ROOT / f"outputs/mpc_dispatch_H{H}.parquet"
    b = bill_from_dispatch(p)
    if b:
        controllers.append((f"Our MPC (LightGBM v1, H={H})", b))

# Walkforward MPC
for H in [96]:
    p = ROOT / f"outputs/mpc_walkforward_H{H}.parquet"
    b = bill_from_dispatch(p)
    if b:
        controllers.append((f"Our MPC (Bagging walkforward, H={H})", b))

bill_rows = []
for name, per in controllers:
    a = per["April"]["net_bill"]
    s = per["September"]["net_bill"]
    bill_rows.append({
        "controller": name,
        "April_EUR":     round(a, 2),
        "September_EUR": round(s, 2),
        "Total_EUR":     round(a + s, 2),
    })

bill_df = pd.DataFrame(bill_rows)
A_total = bill_df[bill_df["controller"].str.startswith("Baseline A")]["Total_EUR"].iloc[0]
bill_df["Savings_vs_A_EUR"] = (A_total - bill_df["Total_EUR"]).round(2)
bill_df["Savings_vs_A_pct"] = (bill_df["Savings_vs_A_EUR"] / abs(A_total) * 100).round(1)

print("\n\nBILL RESULTS (April + September 2025)")
print(bill_df.to_string(index=False))
bill_df.to_csv(SUB / "data/bills_and_savings.csv", index=False)

# ── Combined master report CSV ─────────────────────────────
print("\nWriting combined master CSV...")
with open(SUB / "data/master_results.csv", "w", encoding="utf-8") as f:
    f.write("=== SOLSHIP HACKATHON 2026 — MASTER RESULTS ===\n")
    f.write("Eval window: April 2025 + September 2025\n")
    f.write("Battery: 16 kWh / +-8 kW   Grid: +-6 kW   Round-trip eff: 90%\n")
    f.write("\n\n# 1. FORECAST QUALITY (Apr+Sep 2025 test)\n")
    forecast_df.to_csv(f, index=False)
    f.write("\n\n# 2. CONTROLLER BILLS\n")
    bill_df.to_csv(f, index=False)
    f.write("\n\n# Notes\n")
    f.write("- Baseline A is the existing on-site controller (EUR -7.44 reference).\n")
    f.write("- Oracle is LP optimization with perfect foresight at H=96.\n")
    f.write("- Our best MPC is bagging walkforward (trained with up-to-test-month data per supervisor's allowance).\n")
    f.write("- Forecast NRMSE = RMSE / mean(load_kw) * 100; mean(load Apr+Sep 2025) = 0.901 kW.\n")

print(f"Saved -> {SUB/'data'}")
