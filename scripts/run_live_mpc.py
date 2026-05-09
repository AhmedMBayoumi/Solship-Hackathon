"""
Run rolling-horizon MPC with live inference (actual lags at each step).
Tests multiple horizons and reports bill vs baselines.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

ROOT = Path(__file__).parents[1]

from src.controller.live_forecaster import make_live_factory
from src.controller.mpc_loop import run_both_months

# Load full 2025 from features_all — includes pre-month context for lag_96, lag_672
df_all  = pd.read_parquet(ROOT / "data/features/features_all.parquet")
df_2025 = df_all[df_all["timestamp"].dt.year == 2025].copy().sort_values("timestamp").reset_index(drop=True)

BASELINE_A = -7.57
ORACLE     = -20.14

factory = make_live_factory(df_2025)

horizons = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [4, 8, 16, 24, 48, 96]

print(f"{'H':>6}  {'Net Bill':>10}  {'vs A':>8}  {'Time':>6}")
print("-" * 40)

results = {}
for H in horizons:
    t0 = time.time()
    result, total = run_both_months(df_2025, factory, H=H, label=f"live-H{H}")
    elapsed = time.time() - t0
    bill = total["net_bill"]
    vs_a = bill - BASELINE_A
    results[H] = bill
    print(f"{H:>6}  {bill:>+10.2f}  {vs_a:>+8.2f}  {elapsed:>5.1f}s")

print()
print(f"Baseline A : EUR {BASELINE_A:+.2f}")
print(f"Oracle     : EUR {ORACLE:+.2f}")

# Save best result
best_H = min(results, key=results.get)
print(f"Best H     : {best_H}  ->  EUR {results[best_H]:+.2f}")

out = ROOT / "outputs" / f"live_mpc_H{best_H}.parquet"
factory2 = make_live_factory(df_2025)
res_df, _ = run_both_months(df_2025, factory2, H=best_H, label=f"live-H{best_H}-save")
res_df.to_parquet(out, index=False)
print(f"Saved -> {out}")
