"""Run MPC with rolling-weekly forecast (current BEST 60.75%)."""
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
preds = pd.read_csv(ROOT / "outputs/forecasts/rolling_weekly_test_preds.csv", parse_dates=["timestamp"])

def factory(df_month):
    merged = df_month[["timestamp"]].merge(preds, on="timestamp", how="left")
    lp = merged["load_pred"].values.astype(float)
    for i in range(len(lp)):
        if np.isnan(lp[i]): lp[i] = lp[i-1] if i > 0 else 1.0
    def fn(t, H_in):
        end = min(t + H_in, len(lp))
        return lp[t:end]
    return fn

print("MPC with rolling-weekly forecast (60.75% NRMSE)\n")
print(f"  {'H':>3}  {'April':>8} {'Sept':>8} {'Total':>8} {'vs A':>7} {'sec':>5}")
print("  " + "-" * 50)

results = []
for H in [4, 24, 48, 96]:
    t0 = time.time()
    res, total = run_both_months(df_2025, lambda dm: factory(dm), H=H, verbose=False)
    bills = []
    for m in [4, 9]:
        sub = res[res["timestamp"].dt.month == m]
        bills.append(compute_bill(sub, sub["p_grid_kw"])["net_bill"])
    elapsed = time.time() - t0
    vs_a = total["net_bill"] - (-7.57)
    print(f"  {H:>3}  {bills[0]:>+8.2f} {bills[1]:>+8.2f} {total['net_bill']:>+8.2f} {vs_a:>+7.2f} {elapsed:>5.1f}")
    results.append({"H": H, "april": bills[0], "sept": bills[1], "total": total["net_bill"]})
    if H == 96:
        out = ROOT / "outputs" / "mpc_rolling_weekly_H96.parquet"
        res.to_parquet(out, index=False)
        print(f"      saved -> {out}")

print(f"\n  Baseline A : EUR -7.57")
print(f"  Oracle H=96: EUR -20.14")
new_h96 = next(r for r in results if r["H"] == 96)
print(f"\nProgression at H=96:")
print(f"  Heavy reg walkforward    : EUR -18.79  (NRMSE 61.46%)")
print(f"  Light reg walkforward    : EUR -19.11  (NRMSE 60.83%)")
print(f"  Rolling weekly walkforward (BEST): EUR {new_h96['total']:+.2f}  (NRMSE 60.75%)")
