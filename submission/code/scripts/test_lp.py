import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from src.controller.lp_optimizer import solve_horizon

# Quick sanity test: solve a 4-step horizon
load = np.array([1.5, 2.0, 1.0, 1.5])
pv   = np.array([0.0, 0.0, 0.0, 0.0])
buy  = np.array([0.254, 0.254, 0.244, 0.244])
sell = np.array([0.11, 0.11, 0.11, 0.11])
soc_init = 0.5

p_bat, soc_next = solve_horizon(load, pv, buy, sell, soc_init, H=4)
print(f"LP test OK: p_bat={p_bat:.4f} kW, soc_next={soc_next:.4f}")
