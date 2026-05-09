"""
Run oracle through the FIXED mpc_loop with several horizons.
This is the true upper bound (perfect foresight) using the same
energy-balance logic as our MPC.
"""
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


def make_oracle_factory():
    def factory(df_month):
        load_actual = df_month["load_kw"].values
        def fn(t, H):
            end = min(t + H, len(load_actual))
            return load_actual[t:end]
        return fn
    return factory


print(f"Oracle via fixed MPC loop:")
print(f"  {'H':>6} {'April':>8} {'Sept':>8} {'Total':>8} {'sec':>6}")
print("  " + "-" * 40)

for H in [24, 48, 96, 192, 288, 672]:
    t0 = time.time()
    factory = make_oracle_factory()
    res, total = run_both_months(df_2025, factory, H=H, verbose=False)
    bills = []
    for m in [4, 9]:
        sub = res[res["timestamp"].dt.month == m]
        b = compute_bill(sub, sub["p_grid_kw"])
        bills.append(b["net_bill"])
    print(f"  {H:>6} {bills[0]:>+8.2f} {bills[1]:>+8.2f} {total['net_bill']:>+8.2f} {time.time()-t0:>5.1f}")
