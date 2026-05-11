"""Run MPC with our NEW BEST forecast: final_blend_test_preds.csv (60.41% NRMSE)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from src.controller.mpc_loop import run_both_months
from src.eval.compute_bill import compute_bill

ROOT = Path(__file__).parents[1]

df_2025 = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df_2025[df_2025["timestamp"].dt.year == 2025].copy().reset_index(drop=True)

PRED_FILE = ROOT / "outputs/forecasts/final_blend_test_preds.csv"
preds = pd.read_csv(PRED_FILE, parse_dates=["timestamp"])
print(f"Forecast file: {PRED_FILE.name}")
print(f"  rows: {len(preds)}  cols: {list(preds.columns)}")

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

print("\n" + "=" * 60)
print("MPC with 60.41% blend forecast (causal — real-time deployable)")
print("=" * 60)
print(f"  {'H':>3}  {'April':>8} {'Sept':>8} {'Total':>8} {'vs A':>7} {'sec':>5}")
print("  " + "-" * 50)

results = []
for H in [4, 8, 16, 24, 48, 96]:
    t0 = time.time()
    res, total = run_both_months(df_2025, make_factory(preds), H=H, verbose=False)
    bills = []
    for m in [4, 9]:
        sub = res[res["timestamp"].dt.month == m]
        b = compute_bill(sub, sub["p_grid_kw"])
        bills.append(b["net_bill"])
    elapsed = time.time() - t0
    vs_a = total["net_bill"] - (-7.57)
    print(f"  {H:>3}  {bills[0]:>+8.2f} {bills[1]:>+8.2f} {total['net_bill']:>+8.2f} {vs_a:>+7.2f} {elapsed:>5.1f}")
    results.append({"H": H, "april": bills[0], "sept": bills[1], "total": total["net_bill"],
                    "vs_a": vs_a, "time": elapsed})
    if H == 96:
        out = ROOT / "outputs" / "mpc_blend_H96.parquet"
        res.to_parquet(out, index=False)
        print(f"      saved -> {out}")

print(f"\n  Baseline A (existing controller): EUR -7.57")
print(f"  Oracle (perfect foresight, H=96):  EUR -20.14")

# Compare to previous best (60.83% bagging walkforward submission)
prev_h96 = -19.16  # last reported value with bagging_walkforward_FINAL forecast
new_h96  = next(r for r in results if r["H"] == 96)
print(f"\nProgression at H=96 (1-day battery cycle):")
print(f"  Bagging walkforward FINAL (60.83% NRMSE): EUR {prev_h96:+.2f}")
print(f"  Final blend (60.41% NRMSE, this run)    : EUR {new_h96['total']:+.2f}")
print(f"  Improvement: EUR {new_h96['total'] - prev_h96:+.2f}")

pd.DataFrame(results).to_csv(ROOT / "outputs/horizon_sweep_blend.csv", index=False)
print(f"\nHorizon sweep saved -> outputs/horizon_sweep_blend.csv")
