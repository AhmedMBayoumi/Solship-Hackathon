"""Run rolling-horizon MPC with current best forecast."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from src.controller.mpc_loop import run_both_months
from src.eval.compute_bill import compute_bill

ROOT = Path(__file__).parents[1]

H = int(sys.argv[1]) if len(sys.argv) > 1 else 96

# Load 2025 data
df_2025 = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df_2025[df_2025["timestamp"].dt.year == 2025].copy().reset_index(drop=True)

# Load predictions
pred_file = ROOT / "outputs" / "forecasts" / "ensemble_test_preds.csv"
if not pred_file.exists():
    pred_file = ROOT / "outputs" / "forecasts" / "lgbm_test_preds.csv"
    print(f"Using LightGBM predictions (ensemble not ready)")
else:
    print(f"Using ensemble predictions")

preds_df = pd.read_csv(pred_file, parse_dates=["timestamp"])
print(f"Predictions loaded: {len(preds_df)} rows from {pred_file.name}")

def make_factory(preds_df):
    def factory(df_month):
        merged = df_month[["timestamp"]].merge(preds_df, on="timestamp", how="left")
        load_pred = merged["load_pred"].values.astype(float)
        for i in range(len(load_pred)):
            if np.isnan(load_pred[i]):
                load_pred[i] = load_pred[i-1] if i > 0 else 1.5
        def forecast_fn(t, H_inner):
            end = min(t + H_inner, len(load_pred))
            return load_pred[t:end]
        return forecast_fn
    return factory

print(f"\nRunning MPC with H={H} ({H*15} min horizon)...")
t0 = time.time()
result, total = run_both_months(df_2025, make_factory(preds_df), H=H, label=f"H={H}")
elapsed = time.time() - t0

out = ROOT / "outputs" / f"mpc_dispatch_H{H}.parquet"
result.to_parquet(out, index=False)

print(f"\n{'='*50}")
print(f"RESULTS SUMMARY (H={H})")
print(f"{'='*50}")
print(f"Baseline A   : EUR -7.57")
print(f"Oracle       : EUR -20.14")
print(f"Our MPC H={H}: EUR {total['net_bill']:+.2f}")
savings_vs_A = -7.57 - total["net_bill"]
savings_pct_A = savings_vs_A / abs(-7.57) * 100
oracle_gap = total["net_bill"] - (-20.14)
print(f"Savings vs A : EUR {savings_vs_A:+.2f} ({savings_pct_A:+.1f}%)")
print(f"Oracle gap   : EUR {oracle_gap:+.2f}")
print(f"Time         : {elapsed:.1f}s")
