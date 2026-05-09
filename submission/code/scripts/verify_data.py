"""
Sanity checks on data, code, and dispatches.
Verifies: energy balance, sign conventions, SOC bounds, grid bounds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

print("=" * 60)
print("DATA INTEGRITY CHECKS")
print("=" * 60)

# 1. Raw vs processed dataset
proc = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
print(f"\n1. dataset_processed.csv:")
print(f"   Rows: {len(proc)}  cols: {list(proc.columns)}")
print(f"   Date range: {proc['timestamp'].min()} -> {proc['timestamp'].max()}")
print(f"   load_kw  range: [{proc['load_kw'].min():.3f}, {proc['load_kw'].max():.3f}]")
print(f"   pv_kw    range: [{proc['pv_kw'].min():.3f}, {proc['pv_kw'].max():.3f}]")
print(f"   p_battery range: [{proc['p_battery_kw'].min():.3f}, {proc['p_battery_kw'].max():.3f}]")
print(f"   grid_kw  range: [{proc['grid_kw'].min():.3f}, {proc['grid_kw'].max():.3f}]")
print(f"   buy_price uniq: {sorted(proc['buy_price'].unique())}")
print(f"   tariff_band   : {proc['tariff_band'].value_counts().to_dict()}")

# 2. Verify dataset energy balance: load = pv + battery + grid
err = (proc["load_kw"] - proc["pv_kw"] - proc["p_battery_kw"] - proc["grid_kw"])
print(f"\n2. Dataset energy balance (load - pv - p_bat - grid):")
print(f"   |max|: {err.abs().max():.6f}  rms: {(err**2).mean()**0.5:.6f}")
if err.abs().max() < 1e-3:
    print("   PASS - energy balance holds in raw data")

# 3. Tariff band consistency: F1/F2/F3 hours
print(f"\n3. Tariff band logic check (sample):")
sample = proc.sample(20, random_state=0).sort_values("timestamp")
for _, r in sample.iterrows():
    h = r["timestamp"].hour
    dow = r["timestamp"].dayofweek  # 0=Mon
    print(f"   {r['timestamp']}  dow={dow}  h={h}  band={r['tariff_band']}  buy={r['buy_price']:.4f}  hol={r['is_holiday']}")

# 4. Check our MPC dispatch
for H in [96, 192]:
    p = ROOT / f"outputs/mpc_dispatch_H{H}.parquet"
    if not p.exists():
        continue
    print(f"\n4. MPC dispatch H={H}:")
    res = pd.read_parquet(p)
    err = res["load_kw"] - res["pv_kw"] - res["p_battery_kw"] - res["p_grid_kw"]
    print(f"   Energy balance error: max|err|={err.abs().max():.4f} kW   rms={(err**2).mean()**0.5:.4f}")
    # Bound checks
    p_bat_violations = ((res["p_battery_kw"].abs() > 8.001).sum())
    p_g_violations   = ((res["p_grid_kw"].abs()    > 6.001).sum())
    soc_violations   = ((res["soc"] < -0.001) | (res["soc"] > 1.001)).sum()
    print(f"   |p_battery| > 8 : {p_bat_violations} timesteps")
    print(f"   |p_grid|    > 6 : {p_g_violations} timesteps")
    print(f"   SoC out of [0,1]: {soc_violations} timesteps")

# 5. Baseline A check
ba = proc[proc["timestamp"].dt.year == 2025].copy()
ba_err = ba["load_kw"] - ba["pv_kw"] - ba["p_battery_kw"] - ba["grid_kw"]
print(f"\n5. Baseline A (2025) energy balance error: max|err|={ba_err.abs().max():.4f}")
print(f"   p_battery range: [{ba['p_battery_kw'].min():.3f}, {ba['p_battery_kw'].max():.3f}]")
print(f"   grid_kw range  : [{ba['grid_kw'].min():.3f}, {ba['grid_kw'].max():.3f}]")

# 6. Feature parquet integrity
ft = pd.read_parquet(ROOT / "data/features/features_test.parquet")
print(f"\n6. features_test.parquet: {len(ft)} rows, {len(ft.columns)} cols")
print(f"   Date range: {ft['timestamp'].min()} -> {ft['timestamp'].max()}")
ft_2025 = ft[ft["timestamp"].dt.year == 2025]
print(f"   2025 months: {sorted(ft_2025['timestamp'].dt.month.unique())}")
nas = ft.isna().sum()
big_nas = nas[nas > 0]
print(f"   Columns with NaN: {dict(big_nas) if len(big_nas) > 0 else 'none'}")
