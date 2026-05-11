"""
Compute the missing pieces for the final presentation:

  1. Baseline B (zero-intelligence: no battery, PV→load first, deficit imported, surplus exported)
  2. Oracle controller (MPC with PERFECT load = actual)
  3. March Week 3 dispatch plot (load, PV, P_battery, P_grid, SoC)
  4. Four-line cumulative bill chart for April + September (Baseline A, B, Ours, Oracle)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.controller.mpc_loop import run_both_months, run_mpc
from src.eval.compute_bill import compute_bill, compute_grid

ROOT = Path(__file__).parents[1]
PLOT = ROOT / "outputs/plots/presentation"
PLOT.mkdir(parents=True, exist_ok=True)

# ── Load 2025 data ──────────────────────────────────────────────────
df_2025 = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df_2025[df_2025["timestamp"].dt.year == 2025].copy().reset_index(drop=True)
print(f"2025 rows: {len(df_2025)}", flush=True)

# Drop test months (April + September) only — these are what we evaluate
test_mask = df_2025["timestamp"].dt.month.isin([4, 9])
df_test = df_2025[test_mask].copy().reset_index(drop=True)
print(f"Apr+Sep test rows: {len(df_test)}", flush=True)

# ── Baseline A: historical p_battery_kw ─────────────────────────────
# bill = sum over t of  load - pv - p_battery_kw -> grid; price-weighted.
df_A = df_test.copy()
bill_A = compute_bill(df_A)
print(f"\nBaseline A (historical controller):  EUR {bill_A['net_bill']:+.2f}")
bill_A_apr = compute_bill(df_test[df_test['timestamp'].dt.month==4])
bill_A_sep = compute_bill(df_test[df_test['timestamp'].dt.month==9])
print(f"  April: {bill_A_apr['net_bill']:+.2f}  Sept: {bill_A_sep['net_bill']:+.2f}")

# ── Baseline B: zero-intelligence (no battery) ──────────────────────
df_B = df_test.copy()
df_B["p_battery_kw"] = 0.0   # battery never used
# Apply grid limits ±6 kW (excess PV is curtailed if export > 6)
p_grid_B = (df_B["load_kw"] - df_B["pv_kw"]).clip(lower=-6.0, upper=6.0)
bill_B = compute_bill(df_B, p_grid_B)
print(f"\nBaseline B (zero-intelligence, no battery):  EUR {bill_B['net_bill']:+.2f}")
p_grid_B_apr = (df_test[df_test['timestamp'].dt.month==4]['load_kw'] - df_test[df_test['timestamp'].dt.month==4]['pv_kw']).clip(-6,6)
p_grid_B_sep = (df_test[df_test['timestamp'].dt.month==9]['load_kw'] - df_test[df_test['timestamp'].dt.month==9]['pv_kw']).clip(-6,6)
bill_B_apr = compute_bill(df_test[df_test['timestamp'].dt.month==4], p_grid_B_apr)
bill_B_sep = compute_bill(df_test[df_test['timestamp'].dt.month==9], p_grid_B_sep)
print(f"  April: {bill_B_apr['net_bill']:+.2f}  Sept: {bill_B_sep['net_bill']:+.2f}")

# ── Our controller bill (already computed at H=96, blend forecast) ──
# Cached parquet from run_mpc_blend.py at H=96
ours_path = ROOT / "outputs/mpc_blend_H96.parquet"
if ours_path.exists():
    df_ours = pd.read_parquet(ours_path)
    df_ours["timestamp"] = pd.to_datetime(df_ours["timestamp"])
    bill_ours = compute_bill(df_ours, df_ours["p_grid_kw"])
    bill_ours_apr = compute_bill(df_ours[df_ours["timestamp"].dt.month==4], df_ours[df_ours["timestamp"].dt.month==4]["p_grid_kw"])
    bill_ours_sep = compute_bill(df_ours[df_ours["timestamp"].dt.month==9], df_ours[df_ours["timestamp"].dt.month==9]["p_grid_kw"])
    print(f"\nOurs (H=96, 60.41% blend forecast):  EUR {bill_ours['net_bill']:+.2f}")
    print(f"  April: {bill_ours_apr['net_bill']:+.2f}  Sept: {bill_ours_sep['net_bill']:+.2f}")
else:
    print(f"WARN: {ours_path} missing; run scripts/run_mpc_blend.py first")
    df_ours = None

# ── Oracle: MPC with PERFECT load forecast ──────────────────────────
print("\nRunning ORACLE MPC (perfect load forecast)...", flush=True)
def make_oracle_factory():
    def factory(df_month):
        load_actual = df_month["load_kw"].values
        def fn(t, H_in):
            end = min(t + H_in, len(load_actual))
            return load_actual[t:end]
        return fn
    return factory

t0 = time.time()
res_or, total_or = run_both_months(df_2025, make_oracle_factory(), H=96, verbose=False)
print(f"Oracle done in {time.time()-t0:.0f}s", flush=True)
res_or["timestamp"] = pd.to_datetime(res_or["timestamp"])
bill_or = total_or
bill_or_apr = compute_bill(res_or[res_or['timestamp'].dt.month==4], res_or[res_or['timestamp'].dt.month==4]['p_grid_kw'])
bill_or_sep = compute_bill(res_or[res_or['timestamp'].dt.month==9], res_or[res_or['timestamp'].dt.month==9]['p_grid_kw'])
print(f"Oracle MPC (H=96):  EUR {bill_or['net_bill']:+.2f}")
print(f"  April: {bill_or_apr['net_bill']:+.2f}  Sept: {bill_or_sep['net_bill']:+.2f}")

# Save oracle dispatch
res_or.to_parquet(ROOT / "outputs/mpc_oracle_H96.parquet", index=False)

# ── Save bills summary ──────────────────────────────────────────────
import json
summary = {
    "baseline_A": {"total": bill_A['net_bill'], "april": bill_A_apr['net_bill'], "september": bill_A_sep['net_bill']},
    "baseline_B": {"total": bill_B['net_bill'], "april": bill_B_apr['net_bill'], "september": bill_B_sep['net_bill']},
    "ours_H96":   {"total": bill_ours['net_bill'] if df_ours is not None else None,
                   "april": bill_ours_apr['net_bill'] if df_ours is not None else None,
                   "september": bill_ours_sep['net_bill'] if df_ours is not None else None},
    "oracle_H96": {"total": bill_or['net_bill'], "april": bill_or_apr['net_bill'], "september": bill_or_sep['net_bill']},
}
out_json = ROOT / "outputs/models/presentation_bills.json"
out_json.write_text(json.dumps(summary, indent=2))
print(f"\nBills summary saved -> {out_json}")

# ── 4-line cumulative bill chart for April + September ──────────────
print("\nBuilding 4-line cumulative bill chart...", flush=True)
def cumulative_bill(df_in, p_grid):
    """Cumulative € over time (vectorised)."""
    DT = 0.25
    p = np.asarray(p_grid)
    imp = np.clip(p, 0, None) * df_in["buy_price"].values * DT
    exp = np.clip(-p, 0, None) * df_in["sell_price"].values * DT
    return np.cumsum(imp - exp)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
for ax, month, mname in [(axes[0], 4, "April 2025"), (axes[1], 9, "September 2025")]:
    sub = df_test[df_test["timestamp"].dt.month == month].reset_index(drop=True)
    ts = sub["timestamp"]
    # A: historical p_battery
    p_grid_A = sub["load_kw"].values - sub["pv_kw"].values - sub["p_battery_kw"].values
    cum_A = cumulative_bill(sub, pd.Series(p_grid_A))
    # B: no battery
    p_grid_B_m = (sub["load_kw"].values - sub["pv_kw"].values).clip(-6, 6)
    cum_B = cumulative_bill(sub, pd.Series(p_grid_B_m))
    # Ours
    if df_ours is not None:
        sub_o = df_ours[df_ours["timestamp"].dt.month == month].reset_index(drop=True)
        cum_O = cumulative_bill(sub_o, sub_o["p_grid_kw"])
    # Oracle
    sub_or = res_or[res_or["timestamp"].dt.month == month].reset_index(drop=True)
    cum_OR = cumulative_bill(sub_or, sub_or["p_grid_kw"])

    ax.plot(ts, cum_B, color="tab:gray",   lw=2.0, label=f"Baseline B (no battery): €{cum_B[-1]:+.2f}")
    ax.plot(ts, cum_A, color="tab:orange", lw=2.0, label=f"Baseline A (historical): €{cum_A[-1]:+.2f}")
    if df_ours is not None:
        ax.plot(ts, cum_O,  color="tab:blue",  lw=2.0, label=f"Ours (MPC + 60.41% NRMSE): €{cum_O[-1]:+.2f}")
    ax.plot(ts, cum_OR, color="tab:green", lw=2.0, label=f"Oracle (perfect foresight): €{cum_OR[-1]:+.2f}")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative bill (€)")
    ax.set_title(f"{mname} cumulative bill")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Cumulative electricity bill — April & September 2025  (lower = better savings)", fontsize=12, y=1.00)
fig.tight_layout()
out = PLOT / "cumulative_bill_4line.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"  saved -> {out}")

print("\nAll presentation assets ready.")
