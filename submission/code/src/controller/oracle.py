"""
Perfect-foresight oracle: runs LP with actual 2025 load.
This is the upper bound — no forecast error.
Compare our MPC bill vs oracle bill to quantify the forecast error cost.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.controller.lp_optimizer import solve_horizon, CAPACITY, EFF, DT, GRID_MAX, POWER_MAX
from src.eval.compute_bill import compute_bill


SOC_INIT = 0.5


def run_oracle_month(df_month: pd.DataFrame, H: int = 96, verbose: bool = True, label: str = "") -> tuple:
    """Run oracle MPC with actual load (no forecast error)."""
    n = len(df_month)
    soc = SOC_INIT

    timestamps, load_list, pv_list, p_bat_list, p_grid_list, soc_list = [], [], [], [], [], []

    for t in range(n):
        H_eff = min(H, n - t)
        load_true = df_month["load_kw"].values[t : t + H_eff]  # perfect knowledge
        pv        = df_month["pv_kw"].values[t : t + H_eff]
        buy_win   = df_month["buy_price"].values[t : t + H_eff]
        sell_win  = df_month["sell_price"].values[t : t + H_eff]

        p_bat, _ = solve_horizon(load_true, pv, buy_win, sell_win, soc, H_eff)

        # Enforce constraints
        p_bat = float(np.clip(p_bat, -8.0, 8.0))
        if p_bat > 0:
            p_bat = min(p_bat, soc * CAPACITY * EFF / DT, 8.0)
        else:
            p_bat = max(p_bat, -(1.0 - soc) * CAPACITY / (EFF * DT), -8.0)

        row   = df_month.iloc[t]
        p_g   = row["load_kw"] - row["pv_kw"] - p_bat
        p_g   = float(np.clip(p_g, -GRID_MAX, GRID_MAX))

        if p_bat < 0:
            soc = soc + abs(p_bat) * EFF * DT / CAPACITY
        else:
            soc = soc - p_bat / EFF * DT / CAPACITY
        soc = float(np.clip(soc, 0.0, 1.0))

        timestamps.append(row["timestamp"])
        load_list.append(row["load_kw"])
        pv_list.append(row["pv_kw"])
        p_bat_list.append(p_bat)
        p_grid_list.append(p_g)
        soc_list.append(soc)

    result = pd.DataFrame({
        "timestamp":    timestamps,
        "load_kw":      load_list,
        "pv_kw":        pv_list,
        "p_battery_kw": p_bat_list,
        "p_grid_kw":    p_grid_list,
        "soc":          soc_list,
        "buy_price":    df_month["buy_price"].values,
        "sell_price":   df_month["sell_price"].values,
    })

    bill = compute_bill(result, result["p_grid_kw"])
    if verbose:
        pfx = f"[Oracle {label}] "
        print(f"{pfx}H={H} | Net bill: EUR {bill['net_bill']:+.2f}"
              f"  (import {bill['import_cost']:.2f}, export {bill['export_revenue']:.2f})")
    return result, bill


def run_oracle(df_2025: pd.DataFrame, H: int = 96) -> dict:
    """Oracle on both April and September 2025."""
    total = {"import_cost": 0, "export_revenue": 0, "net_bill": 0}
    results = []
    for month, name in [(4, "April"), (9, "September")]:
        df_m = df_2025[df_2025["timestamp"].dt.month == month].copy().reset_index(drop=True)
        res, bill = run_oracle_month(df_m, H=H, label=name)
        for k in total:
            total[k] = round(total[k] + bill[k], 4)
        results.append(res)
    combined = pd.concat(results, ignore_index=True)
    print(f"\n[Oracle] TOTAL Apr+Sep: EUR {total['net_bill']:+.2f}")
    print(f"[Oracle] vs Baseline A (EUR -7.57): potential savings EUR {-7.57 - total['net_bill']:+.2f}")
    return combined, total
