"""
Compare what Baseline A does vs what our MPC does.
Look at battery cycling, energy throughput, and dispatch patterns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

df = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df[df["timestamp"].dt.year == 2025].copy()

# Baseline A uses actual p_battery_kw column
DT = 0.25
for mo, name in [(4, "April"), (9, "September")]:
    sub = df_2025[df_2025["timestamp"].dt.month == mo]
    p_b   = sub["p_battery_kw"].values
    load  = sub["load_kw"].values
    pv    = sub["pv_kw"].values
    p_g   = load - pv - p_b

    chg_kwh = np.sum(np.maximum(-p_b, 0)) * DT     # energy charged (kWh)
    dis_kwh = np.sum(np.maximum(p_b, 0)) * DT      # energy discharged
    imp_kwh = np.sum(np.maximum(p_g, 0)) * DT       # energy imported
    exp_kwh = np.sum(np.maximum(-p_g, 0)) * DT      # energy exported
    load_kwh = load.sum() * DT
    pv_kwh   = pv.sum() * DT

    print(f"\n=== {name} 2025 — Baseline A ===")
    print(f"  Load total      : {load_kwh:.1f} kWh   mean = {load.mean():.3f} kW")
    print(f"  PV total        : {pv_kwh:.1f} kWh   mean = {pv.mean():.3f} kW")
    print(f"  Battery charged : {chg_kwh:.1f} kWh")
    print(f"  Battery discharged: {dis_kwh:.1f} kWh   (round-trip = {dis_kwh/chg_kwh*100:.1f}%)")
    print(f"  Grid imported   : {imp_kwh:.1f} kWh")
    print(f"  Grid exported   : {exp_kwh:.1f} kWh")
    print(f"  Battery max p   : charge {-p_b.min():.2f}  discharge {p_b.max():.2f}")
    print(f"  Time discharging: {(p_b > 0.1).sum()/len(p_b)*100:.1f}% of timesteps")
    print(f"  Time charging   : {(p_b < -0.1).sum()/len(p_b)*100:.1f}% of timesteps")
    print(f"  Time idle       : {(np.abs(p_b) < 0.1).sum()/len(p_b)*100:.1f}% of timesteps")

# Compare to our MPC
print("\n\n=== Our MPC (H=672) ===")
mpc = pd.read_parquet(ROOT / "outputs/mpc_dispatch_H96.parquet")  # might not exist for 672
for mo, name in [(4, "April"), (9, "September")]:
    sub = mpc[mpc["timestamp"].dt.month == mo]
    if len(sub) == 0:
        continue
    p_b = sub["p_battery_kw"].values
    p_g = sub["p_grid_kw"].values
    chg_kwh = np.sum(np.maximum(-p_b, 0)) * DT
    dis_kwh = np.sum(np.maximum(p_b, 0)) * DT
    imp_kwh = np.sum(np.maximum(p_g, 0)) * DT
    exp_kwh = np.sum(np.maximum(-p_g, 0)) * DT
    print(f"\n  {name}:")
    print(f"  Battery charged : {chg_kwh:.1f} kWh")
    print(f"  Battery discharged: {dis_kwh:.1f} kWh")
    print(f"  Grid imported   : {imp_kwh:.1f} kWh")
    print(f"  Grid exported   : {exp_kwh:.1f} kWh")
    print(f"  Time discharging: {(p_b > 0.1).sum()/len(p_b)*100:.1f}%")
    print(f"  Time charging   : {(p_b < -0.1).sum()/len(p_b)*100:.1f}%")
