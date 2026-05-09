"""
Rolling-horizon MPC loop.

At each step t:
  1. Get load forecast for next H steps from the forecast model
  2. Use known PV, buy_price, sell_price from dataset
  3. Solve LP over horizon H
  4. Execute only the first battery action
  5. Update SoC from actual executed action
  6. Advance t+1 and repeat
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.controller.lp_optimizer import solve_horizon, DT, EFF, CAPACITY
from src.eval.compute_bill import compute_bill

SOC_INIT = 0.5   # 50% = 8 kWh


def run_mpc(
    df_month: pd.DataFrame,
    forecast_fn,         # callable(t: int, H: int) -> np.ndarray of shape (H,)
    H: int,
    soc_init: float = SOC_INIT,
    verbose: bool = True,
    label: str = "",
    cycle_penalty: float = 0.0,
) -> pd.DataFrame:
    """
    Run rolling-horizon MPC on a single month DataFrame.

    Parameters
    ----------
    df_month   : DataFrame with columns [timestamp, load_kw, pv_kw, buy_price, sell_price]
    forecast_fn: function(t, H) → np.ndarray load forecast for steps t..t+H-1
    H          : forecast horizon (number of 15-min steps)
    soc_init   : initial SoC fraction (0–1), default 0.5
    """
    n = len(df_month)
    soc = soc_init

    timestamps   = []
    load_actual  = []
    pv_actual    = []
    p_battery    = []
    p_grid_list  = []
    soc_list     = []

    t0 = time.time()
    load_actual_arr = df_month["load_kw"].values
    for t in range(n):
        row = df_month.iloc[t]

        # ── Get forecast ─────────────────────────────────────────────────
        load_fc = forecast_fn(t, H)                        # shape (H,) or shorter at end
        H_eff   = min(H, n - t)
        load_fc = np.asarray(load_fc[:H_eff], dtype=float).copy()
        # Per hackathon rules: at step t we OBSERVE actual current load.
        # Replace the first horizon step with the actual measurement so the
        # current-step decision is not corrupted by forecast noise.
        if H_eff >= 1:
            load_fc[0] = float(load_actual_arr[t])

        pv_win   = df_month["pv_kw"].values[t : t + H_eff]
        buy_win  = df_month["buy_price"].values[t : t + H_eff]
        sell_win = df_month["sell_price"].values[t : t + H_eff]

        # ── Solve LP ─────────────────────────────────────────────────────
        p_bat, soc_lp = solve_horizon(load_fc, pv_win, buy_win, sell_win, soc, H_eff,
                                       cycle_penalty=cycle_penalty)

        # ── Enforce physical constraints on executed action ───────────────
        # Clip to battery power limit
        p_bat = float(np.clip(p_bat, -CAPACITY / DT, CAPACITY / DT))
        p_bat = float(np.clip(p_bat, -8.0, 8.0))

        # Enforce SoC bounds
        if p_bat > 0:  # discharging
            max_dis = soc * CAPACITY * EFF / DT
            p_bat = min(p_bat, max_dis, 8.0)
        else:           # charging
            max_chg = (1.0 - soc) * CAPACITY / (EFF * DT)
            p_bat = max(p_bat, -max_chg, -8.0)

        # ── Compute actual grid from energy balance ───────────────────────
        p_g = row["load_kw"] - row["pv_kw"] - p_bat
        # If grid violates ±6 kW limit, push more onto the battery.
        # NB: p_g = load - pv - p_bat, so increasing p_bat *decreases* p_g.
        if p_g > 6.0:
            delta = p_g - 6.0
            p_bat += delta          # discharge MORE (or charge less) → less import
            # Re-clamp battery + SoC bounds after adjustment
            p_bat = float(np.clip(p_bat, -8.0, 8.0))
            if p_bat > 0:
                p_bat = min(p_bat, soc * CAPACITY * EFF / DT)
            p_g = float(row["load_kw"] - row["pv_kw"] - p_bat)
            p_g = min(p_g, 6.0)     # final hard cap (only triggers if battery couldn't comply)
        elif p_g < -6.0:
            delta = -6.0 - p_g       # positive
            p_bat -= delta          # charge MORE (or discharge less) → less export
            p_bat = float(np.clip(p_bat, -8.0, 8.0))
            if p_bat < 0:
                p_bat = max(p_bat, -(1.0 - soc) * CAPACITY / (EFF * DT))
            p_g = float(row["load_kw"] - row["pv_kw"] - p_bat)
            p_g = max(p_g, -6.0)

        # ── Update SoC from actual battery action ────────────────────────
        if p_bat < 0:   # charging
            soc = soc + abs(p_bat) * EFF * DT / CAPACITY
        else:           # discharging
            soc = soc - p_bat / EFF * DT / CAPACITY
        soc = float(np.clip(soc, 0.0, 1.0))

        timestamps.append(row["timestamp"])
        load_actual.append(row["load_kw"])
        pv_actual.append(row["pv_kw"])
        p_battery.append(p_bat)
        p_grid_list.append(p_g)
        soc_list.append(soc)

    elapsed = time.time() - t0
    result = pd.DataFrame({
        "timestamp":   timestamps,
        "load_kw":     load_actual,
        "pv_kw":       pv_actual,
        "p_battery_kw": p_battery,
        "p_grid_kw":   p_grid_list,
        "soc":         soc_list,
        "buy_price":   df_month["buy_price"].values,
        "sell_price":  df_month["sell_price"].values,
    })

    bill = compute_bill(result, result["p_grid_kw"])
    if verbose:
        pfx = f"[{label}] " if label else ""
        print(f"{pfx}H={H:4d}  |  Net bill: EUR {bill['net_bill']:+.2f}"
              f"  (import {bill['import_cost']:.2f}, export {bill['export_revenue']:.2f})"
              f"  |  {elapsed:.1f}s")
    result["_bill"] = bill["net_bill"]
    return result, bill


def run_both_months(
    df_2025: pd.DataFrame,
    forecast_fn_factory,   # callable(df_month) -> forecast_fn
    H: int,
    verbose: bool = True,
    label: str = "",
    cycle_penalty: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Run MPC on April 2025 and September 2025 independently (separate SoC resets)."""
    results = []
    total_bill = {"import_cost": 0, "export_revenue": 0, "net_bill": 0}

    for month, mname in [(4, "April"), (9, "September")]:
        df_m = df_2025[df_2025["timestamp"].dt.month == month].copy().reset_index(drop=True)
        forecast_fn = forecast_fn_factory(df_m)
        res, bill = run_mpc(df_m, forecast_fn, H, soc_init=SOC_INIT,
                            verbose=verbose, label=f"{label} {mname}",
                            cycle_penalty=cycle_penalty)
        for k in total_bill:
            total_bill[k] = round(total_bill[k] + bill[k], 4)
        results.append(res)

    combined = pd.concat(results, ignore_index=True)
    if verbose:
        pfx = f"[{label}] " if label else ""
        print(f"{pfx}TOTAL Apr+Sep  |  Net bill: EUR {total_bill['net_bill']:+.2f}")
        print(f"{pfx}vs Baseline A (EUR -7.57)  |  Savings: EUR {-7.57 - total_bill['net_bill']:+.2f}")
    return combined, total_bill
