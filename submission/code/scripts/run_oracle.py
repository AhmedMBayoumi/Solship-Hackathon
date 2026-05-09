import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

df = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df[df["timestamp"].dt.year == 2025].copy().reset_index(drop=True)

# Test with just April, first 10 steps
df_apr = df_2025[df_2025["timestamp"].dt.month == 4].copy().reset_index(drop=True)
print(f"April rows: {len(df_apr)}")
print(f"Columns: {list(df_apr.columns[:8])}")

from src.controller.lp_optimizer import solve_horizon
import time

# Test oracle on first 5 steps of April
print("\nTesting oracle on first 5 steps...")
soc = 0.5
for t in range(5):
    H_eff = min(96, len(df_apr) - t)
    load_true = df_apr["load_kw"].values[t : t + H_eff]
    pv        = df_apr["pv_kw"].values[t : t + H_eff]
    buy_win   = df_apr["buy_price"].values[t : t + H_eff]
    sell_win  = df_apr["sell_price"].values[t : t + H_eff]
    try:
        p_bat, soc_next = solve_horizon(load_true, pv, buy_win, sell_win, soc, H_eff)
        print(f"  Step {t}: p_bat={p_bat:.3f} kW, soc={soc_next:.4f}")
        soc = soc_next
    except Exception as e:
        print(f"  Step {t} FAILED: {e}")
        traceback.print_exc()
        break

print("\nLP solve_horizon works. Running full oracle...")
from src.controller.oracle import run_oracle

t0 = time.time()
try:
    result, total = run_oracle(df_2025, H=96)
    elapsed = time.time() - t0
    print(f"\nOracle complete in {elapsed:.0f}s")
    print(f"Baseline A  : EUR -7.57")
    print(f"Oracle bill : EUR {total['net_bill']:+.2f}")
    print(f"Max savings : EUR {-7.57 - total['net_bill']:+.2f}")

    out = ROOT / "outputs" / "oracle_dispatch.parquet"
    result.to_parquet(out, index=False)
    print(f"Saved -> {out}")
except Exception as e:
    print(f"Oracle failed: {e}")
    traceback.print_exc()
