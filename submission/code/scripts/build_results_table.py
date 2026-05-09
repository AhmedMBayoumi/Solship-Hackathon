"""
Generate the final results table for the slide deck (Phase 6.2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from src.eval.compute_bill import compute_bill, baseline_a_bill, baseline_b_bill

ROOT = Path(__file__).parents[1]

df = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df[df["timestamp"].dt.year == 2025].copy()

# Per-month split
def per_month_bill(bill_func, df_):
    out = {}
    for m, n in [(4, "April"), (9, "September")]:
        sub = df_[df_["timestamp"].dt.month == m]
        out[n] = bill_func(sub)
    return out

# Baselines
A_per = per_month_bill(baseline_a_bill, df_2025)
B_per = per_month_bill(baseline_b_bill, df_2025)
print("Baseline A per month:", {k: f"EUR {v['net_bill']:+.2f}" for k,v in A_per.items()})
print("Baseline B per month:", {k: f"EUR {v['net_bill']:+.2f}" for k,v in B_per.items()})

# Our MPC at various H — read saved dispatch parquets
def read_mpc(H):
    p = ROOT / f"outputs/mpc_dispatch_H{H}.parquet"
    if not p.exists():
        return None
    res = pd.read_parquet(p)
    out = {}
    for m, n in [(4, "April"), (9, "September")]:
        sub = res[res["timestamp"].dt.month == m]
        out[n] = compute_bill(sub, sub["p_grid_kw"])
    return out

# Oracle
oracle_p = ROOT / "outputs/oracle_dispatch.parquet"
oracle_per = None
if oracle_p.exists():
    res = pd.read_parquet(oracle_p)
    oracle_per = {}
    for m, n in [(4, "April"), (9, "September")]:
        sub = res[res["timestamp"].dt.month == m]
        oracle_per[n] = compute_bill(sub, sub["p_grid_kw"])

# Our chosen final controllers
H_choices = [96, 192, 672]
mpc_results = {H: read_mpc(H) for H in H_choices}

# Build the table
print()
print("=" * 80)
print("FINAL RESULTS TABLE (April + September 2025)")
print("=" * 80)
print(f"{'Controller':<32} {'April':>10} {'September':>11} {'Total':>10}")
print("-" * 80)

def fmt(v):
    return f"EUR {v:+7.2f}"

# Baselines
a = A_per["April"]["net_bill"]; b = A_per["September"]["net_bill"]
print(f"{'Baseline A (existing controller)':<32} {fmt(a):>10} {fmt(b):>11} {fmt(a+b):>10}")

a = B_per["April"]["net_bill"]; b = B_per["September"]["net_bill"]
print(f"{'Baseline B (no battery)':<32} {fmt(a):>10} {fmt(b):>11} {fmt(a+b):>10}")

# Our MPC
for H in H_choices:
    r = mpc_results[H]
    if r is None:
        print(f"{f'Our MPC (LightGBM, H={H})':<32}  -- not saved --")
        continue
    a = r["April"]["net_bill"]; b = r["September"]["net_bill"]
    print(f"{f'Our MPC (LightGBM, H={H})':<32} {fmt(a):>10} {fmt(b):>11} {fmt(a+b):>10}")

# Oracle
if oracle_per:
    a = oracle_per["April"]["net_bill"]; b = oracle_per["September"]["net_bill"]
    print(f"{'Oracle (perfect foresight, H=96)':<32} {fmt(a):>10} {fmt(b):>11} {fmt(a+b):>10}")

# Savings vs A summary
print()
print("=" * 80)
print("SAVINGS vs Baseline A")
print("=" * 80)
A_total = A_per["April"]["net_bill"] + A_per["September"]["net_bill"]
print(f"  Baseline A total: EUR {A_total:+.2f}")
for H in H_choices:
    r = mpc_results[H]
    if r is None:
        continue
    total = r["April"]["net_bill"] + r["September"]["net_bill"]
    savings = A_total - total
    pct = savings / abs(A_total) * 100 if A_total != 0 else 0
    print(f"  Our MPC H={H:>3}: EUR {total:+.2f}  ->  savings EUR {savings:+.2f}  ({pct:+.1f}% vs A)")

# Save CSV
rows = []
for label, per in [("Baseline A", A_per), ("Baseline B", B_per), ("Oracle (H=96)", oracle_per)]:
    if per is None:
        continue
    rows.append({
        "Controller": label,
        "April":     per["April"]["net_bill"],
        "September": per["September"]["net_bill"],
        "Total":     per["April"]["net_bill"] + per["September"]["net_bill"],
    })
for H in H_choices:
    r = mpc_results[H]
    if r is None:
        continue
    rows.append({
        "Controller": f"Our MPC LGBM H={H}",
        "April":     r["April"]["net_bill"],
        "September": r["September"]["net_bill"],
        "Total":     r["April"]["net_bill"] + r["September"]["net_bill"],
    })
out_csv = ROOT / "outputs/results_table.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nSaved -> {out_csv}")
