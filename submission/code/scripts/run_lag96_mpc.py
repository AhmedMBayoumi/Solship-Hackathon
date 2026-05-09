"""
Run rolling-horizon MPC with lag-96 persistence forecast.
Simple baseline: load[t] ~= load[t-96]  (yesterday at the same time of day).

Often beats ML models when there is distribution shift, because lag_96 is
purely the actual recent observation — no model bias.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

from src.controller.mpc_loop import run_both_months

df_all = pd.read_parquet(ROOT / "data/features/features_all.parquet")
df_2025 = df_all[df_all["timestamp"].dt.year == 2025].copy().sort_values("timestamp").reset_index(drop=True)

BASELINE_A = -7.57
ORACLE     = -20.14


def make_lag96_factory(df_full):
    """Use load[t-96] as the forecast for load[t]."""
    load_full = df_full["load_kw"].values

    def factory(df_month):
        start_ts = df_month["timestamp"].iloc[0]
        offset_idx = df_full.index[df_full["timestamp"] == start_ts]
        offset = int(offset_idx[0]) if len(offset_idx) else 0

        def forecast_fn(t, H):
            n = len(df_month)
            H_eff = min(H, n - t)
            fc = np.zeros(H_eff, dtype=float)
            for k in range(H_eff):
                g = offset + t + k          # global index of step we're forecasting
                lag_g = g - 96
                if lag_g >= 0:
                    fc[k] = float(load_full[lag_g])
                else:
                    fc[k] = 1.0  # fallback — should not happen with full-year context
            return fc

        return forecast_fn

    return factory


horizons = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [4, 24, 96]

print(f"{'H':>6}  {'Net Bill':>10}  {'vs A':>8}  {'Time':>6}")
print("-" * 40)

for H in horizons:
    t0 = time.time()
    factory = make_lag96_factory(df_2025)
    result, total = run_both_months(df_2025, factory, H=H, verbose=False, label=f"lag96-H{H}")
    elapsed = time.time() - t0
    bill = total["net_bill"]
    vs_a = bill - BASELINE_A
    print(f"{H:>6}  {bill:>+10.2f}  {vs_a:>+8.2f}  {elapsed:>5.1f}s")

print()
print(f"Baseline A : EUR {BASELINE_A:+.2f}")
print(f"Oracle     : EUR {ORACLE:+.2f}")
