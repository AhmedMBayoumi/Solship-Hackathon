"""
Diagnose forecast and optimizer:
1. MPC with actual load as forecast (oracle-via-MPC) -- confirms optimizer works
2. MPC with lag_96 persistence -- simple baseline forecast
3. Check LightGBM prediction quality per month
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

df_2025 = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df_2025[df_2025["timestamp"].dt.year == 2025].copy().reset_index(drop=True)

preds = pd.read_csv(ROOT / "outputs/forecasts/lgbm_test_preds.csv", parse_dates=["timestamp"])

# Check NRMSE by month
df_test = df_2025[df_2025["timestamp"].dt.month.isin([4, 9])]
merged = df_test.merge(preds, on="timestamp", how="left")

for month, name in [(4, "April"), (9, "September")]:
    m = merged[merged["timestamp"].dt.month == month]
    rmse = np.sqrt(np.mean((m["load_kw"] - m["load_pred"])**2))
    nrmse = rmse / m["load_kw"].mean() * 100
    mae = np.mean(np.abs(m["load_kw"] - m["load_pred"]))
    print(f"{name} 2025: NRMSE={nrmse:.2f}%  RMSE={rmse:.4f}  MAE={mae:.4f}")
    # Compare to lag_96 persistence
    lag96 = df_2025[df_2025["timestamp"].dt.month.isin([4, 9])].copy()
    lag96["lag_96_pred"] = lag96["load_kw"].shift(96)
    lag96 = lag96[lag96["timestamp"].dt.month == month].dropna()
    rmse_p = np.sqrt(np.mean((lag96["load_kw"] - lag96["lag_96_pred"])**2))
    nrmse_p = rmse_p / lag96["load_kw"].mean() * 100
    print(f"  lag_96 persistence: NRMSE={nrmse_p:.2f}%  RMSE={rmse_p:.4f}")

print()

# Run MPC with actual load (oracle-via-MPC loop), H=96
print("=== Test 1: Oracle-via-MPC (actual load as forecast) ===")
from src.controller.mpc_loop import run_both_months

def oracle_factory(df_month):
    load_actual = df_month["load_kw"].values.copy()
    def forecast_fn(t, H):
        end = min(t + H, len(load_actual))
        return load_actual[t:end]
    return forecast_fn

t0 = time.time()
_, bill_oracle_mpc = run_both_months(df_2025, oracle_factory, H=96, label="Oracle-MPC")
print(f"Time: {time.time()-t0:.1f}s")

print()
print("=== Test 2: lag_96 persistence forecast, H=96 ===")
df_all = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

def lag96_factory(df_month):
    # Get the full sorted series to properly get lag-96
    start_ts = df_month["timestamp"].iloc[0]
    start_idx = df_all[df_all["timestamp"] == start_ts].index[0]
    # Build prediction using actual data from df_all
    def forecast_fn(t, H):
        fc = []
        for k in range(min(H, len(df_month) - t)):
            actual_idx = start_idx + t + k
            lag_idx = actual_idx - 96
            if lag_idx >= 0:
                fc.append(float(df_all.loc[lag_idx, "load_kw"]))
            else:
                fc.append(float(df_month["load_kw"].iloc[max(0, t+k-1)]))
        return np.array(fc)
    return forecast_fn

t0 = time.time()
_, bill_lag96 = run_both_months(df_2025, lag96_factory, H=96, label="lag96-H96")
print(f"Time: {time.time()-t0:.1f}s")

print()
print("=== Summary ===")
print(f"Baseline A      : EUR -7.57")
print(f"Oracle (LP opt) : EUR -20.14")
print(f"Oracle-via-MPC  : EUR {bill_oracle_mpc['net_bill']:+.2f}")
print(f"lag_96 persist  : EUR {bill_lag96['net_bill']:+.2f}")
print(f"LightGBM H=96   : EUR +9.51  (pre-computed)")
