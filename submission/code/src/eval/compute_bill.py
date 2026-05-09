"""
Compute electricity bill for a given dispatch schedule.
Works with any DataFrame that has: timestamp, load_kw, pv_kw, p_battery_kw,
buy_price, sell_price.

Grid power is computed from energy balance:
  p_grid = load_kw - pv_kw - p_battery_kw
  (positive = import, negative = export)
"""
import numpy as np
import pandas as pd


DT = 0.25  # 15-min timestep in hours
EFF = 0.9 ** 0.5   # per-direction efficiency
CAPACITY = 16.0    # kWh


def compute_grid(df: pd.DataFrame) -> pd.Series:
    return df["load_kw"] - df["pv_kw"] - df["p_battery_kw"]


def compute_bill(df: pd.DataFrame, p_grid: pd.Series = None) -> dict:
    if p_grid is None:
        p_grid = compute_grid(df)
    imp = p_grid.clip(lower=0)
    exp = (-p_grid).clip(lower=0)
    import_cost   = (imp * df["buy_price"]  * DT).sum()
    export_revenue= (exp * df["sell_price"] * DT).sum()
    net = import_cost - export_revenue
    return {
        "import_cost":    round(import_cost, 4),
        "export_revenue": round(export_revenue, 4),
        "net_bill":       round(net, 4),
    }


def baseline_a_bill(df_2025: pd.DataFrame) -> dict:
    """Bill using actual p_battery_kw (existing controller)."""
    return compute_bill(df_2025)


def baseline_b_bill(df_2025: pd.DataFrame) -> dict:
    """Bill with no battery (p_battery = 0 always)."""
    df0 = df_2025.copy()
    df0["p_battery_kw"] = 0.0
    return compute_bill(df0)


def reconstruct_soc(p_battery_series: pd.Series, soc_init: float = 0.5,
                    capacity: float = CAPACITY) -> pd.Series:
    soc = [soc_init]
    for pb in p_battery_series.values:
        s = soc[-1]
        if pb < 0:  # charging
            s = s + abs(pb) * EFF * DT / capacity
        else:       # discharging
            s = s - pb / EFF * DT / capacity
        soc.append(np.clip(s, 0.0, 1.0))
    return pd.Series(soc[1:], index=p_battery_series.index)


def print_bill_table(label: str, bill: dict):
    print(f"  {label}:")
    print(f"    Import cost   : EUR {bill['import_cost']:.2f}")
    print(f"    Export revenue: EUR {bill['export_revenue']:.2f}")
    print(f"    Net bill      : EUR {bill['net_bill']:.2f}")
