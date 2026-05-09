"""Run MPC with walkforward bagging forecast at H=96 (1 day = battery cycle)."""
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

preds = pd.read_csv(ROOT / "outputs/forecasts/bagging_walkforward_test_preds.csv", parse_dates=["timestamp"])

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

print(f"Walkforward bagging MPC sweep (per supervisor: H<=96 = 1-day battery cycle)\n")
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
        out = ROOT / "outputs" / f"mpc_walkforward_H{H}.parquet"
        res.to_parquet(out, index=False)
        print(f"      saved -> {out}")

print(f"\n  Baseline A : EUR -7.57")
print(f"  Oracle     : EUR -20.14 (H=96)")
best = min(results, key=lambda r: r["total"])
print(f"\n  Best H = {best['H']}  -> total EUR {best['total']:+.2f}  (vs A: EUR {best['vs_a']:+.2f})")

pd.DataFrame(results).to_csv(ROOT / "outputs/horizon_sweep_walkforward.csv", index=False)
