"""
LP battery dispatch optimizer — scipy.optimize.linprog backend.

Formulated as a standard LP:
  Variables per step t: [p_chg[t], p_dis[t], p_imp[t], p_exp[t]]
  Plus SoC variables:   [soc[0], soc[1], ..., soc[H]]

This avoids the Windows HiGHS crash that occurs when CVXPY calls the
shared library thousands of times inside a tight MPC loop.
scipy.optimize.linprog (HiGHS) is called via C extension, which is stable.
"""
import numpy as np
from scipy.optimize import linprog

DT       = 0.25     # h per timestep
CAPACITY = 16.0     # kWh usable
POWER_MAX= 8.0      # kW battery
GRID_MAX = 6.0      # kW grid
EFF      = 0.9**0.5 # per-direction efficiency (sqrt(0.9))


def solve_horizon(
    load_fc:    np.ndarray,
    pv:         np.ndarray,
    buy_price:  np.ndarray,
    sell_price: np.ndarray,
    soc_init:   float,
    H:          int,
    capacity:   float = CAPACITY,
    power_max:  float = POWER_MAX,
    grid_max:   float = GRID_MAX,
    eff:        float = EFF,
) -> tuple:
    """
    Solve LP over H timesteps.

    Variable layout (5H + 1 total):
      [p_chg_0..p_chg_{H-1},  (indices 0..H-1)
       p_dis_0..p_dis_{H-1},  (indices H..2H-1)
       p_imp_0..p_imp_{H-1},  (indices 2H..3H-1)
       p_exp_0..p_exp_{H-1},  (indices 3H..4H-1)
       soc_0..soc_H]           (indices 4H..5H)

    Returns (p_battery_first, soc_next):
      p_battery_first > 0 → discharge, < 0 → charge
    """
    H = min(H, len(load_fc))
    load_fc    = np.asarray(load_fc[:H], dtype=float)
    pv         = np.asarray(pv[:H],     dtype=float)
    buy_price  = np.asarray(buy_price[:H],  dtype=float)
    sell_price = np.asarray(sell_price[:H], dtype=float)

    n = 5 * H + 1  # total variables

    # Index helpers
    def ic(t): return t           # p_chg[t]
    def id(t): return H + t       # p_dis[t]
    def ii(t): return 2*H + t     # p_imp[t]
    def ie(t): return 3*H + t     # p_exp[t]
    def is_(t): return 4*H + t    # soc[t]

    # ── Objective ────────────────────────────────────────────────────────
    c = np.zeros(n)
    for t in range(H):
        c[ii(t)] =  buy_price[t]  * DT   # import cost (positive, minimise)
        c[ie(t)] = -sell_price[t] * DT   # export revenue (negative, minimise neg)

    # ── Bounds ───────────────────────────────────────────────────────────
    bounds = (
        [(0, power_max)] * H +   # p_chg
        [(0, power_max)] * H +   # p_dis
        [(0, grid_max)]  * H +   # p_imp
        [(0, grid_max)]  * H +   # p_exp
        [(0, 1.0)]       * (H+1) # soc[0..H]
    )

    # ── Equality constraints ──────────────────────────────────────────────
    # 1. Power balance: p_imp[t] + p_dis[t] + pv[t] = load[t] + p_chg[t] + p_exp[t]
    #    => -p_chg[t] + p_dis[t] + p_imp[t] - p_exp[t] = load[t] - pv[t]
    # 2. SoC dynamics: soc[t+1] = soc[t] + (p_chg[t]*eff - p_dis[t]/eff)*DT/C
    #    => soc[t+1] - soc[t] - p_chg[t]*eff*DT/C + p_dis[t]/eff*DT/C = 0
    # 3. Initial SoC: soc[0] = soc_init

    n_eq = H + H + 1
    A_eq = np.zeros((n_eq, n))
    b_eq = np.zeros(n_eq)

    for t in range(H):
        # Power balance
        row = t
        A_eq[row, ic(t)] = -1.0
        A_eq[row, id(t)] =  1.0
        A_eq[row, ii(t)] =  1.0
        A_eq[row, ie(t)] = -1.0
        b_eq[row] = load_fc[t] - pv[t]

        # SoC dynamics
        row = H + t
        A_eq[row, ic(t)]   = -eff * DT / capacity
        A_eq[row, id(t)]   =  (1.0/eff) * DT / capacity
        A_eq[row, is_(t)]  = -1.0
        A_eq[row, is_(t+1)]=  1.0
        b_eq[row] = 0.0

    # Initial SoC
    A_eq[2*H, is_(0)] = 1.0
    b_eq[2*H] = soc_init

    # ── Solve ─────────────────────────────────────────────────────────────
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.status != 0:
        return 0.0, float(soc_init)

    x = result.x
    first_dis = x[id(0)]
    first_chg = x[ic(0)]
    p_bat_first = float(first_dis - first_chg)

    if p_bat_first < 0:  # charging
        soc_next = soc_init + abs(p_bat_first) * eff * DT / capacity
    else:                # discharging
        soc_next = soc_init - p_bat_first / eff * DT / capacity
    soc_next = float(np.clip(soc_next, 0.0, 1.0))

    return p_bat_first, soc_next
