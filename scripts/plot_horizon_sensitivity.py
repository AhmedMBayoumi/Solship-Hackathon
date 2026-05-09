"""
Plot horizon sensitivity (Extension 1, +5 pts).
Shows total bill vs H on log-x axis, with oracle and Baseline A reference lines.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]

# Load sweep results
fname = sys.argv[1] if len(sys.argv) > 1 else "lgbm"
sweep = pd.read_csv(ROOT / f"outputs/horizon_sweep_{fname}.csv")

# Oracle reference (smaller-H sweep done separately, embed values)
oracle_data = {
    24:  6.35,  48: -18.27,  96: -20.14,
    192: -23.45, 288: -24.45, 672: -24.84,
}

BASELINE_A = -7.57

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Bill vs H
ax.plot(sweep["H"], sweep["total"], "o-", label=f"Our MPC ({fname})", color="#0066cc", lw=2)
ax.plot(list(oracle_data.keys()), list(oracle_data.values()), "s--",
        label="Oracle (perfect foresight)", color="green", lw=2)
ax.axhline(BASELINE_A, color="red", ls="--", lw=2, label=f"Baseline A (EUR {BASELINE_A:+.2f})")
ax.set_xscale("log")
ax.set_xlabel("Horizon H (timesteps × 15 min)")
ax.set_ylabel("Net bill April + September 2025 (EUR)")
ax.set_title(f"Bill vs MPC horizon — {fname}")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")

# Annotate best
best_idx = sweep["total"].idxmin()
best_H = int(sweep.iloc[best_idx]["H"])
best_b = sweep.iloc[best_idx]["total"]
ax.annotate(
    f"H={best_H}\nEUR {best_b:+.2f}",
    xy=(best_H, best_b),
    xytext=(best_H * 1.5, best_b - 4),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black"),
)

# Time vs H
ax2.plot(sweep["H"], sweep["time"], "o-", color="#cc6600", lw=2)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Horizon H (timesteps)")
ax2.set_ylabel("Wall time (s) for full Apr+Sep run")
ax2.set_title("Compute time vs horizon")
ax2.grid(True, which="both", alpha=0.3)

# Find knee: smallest H within 1% of best total
within_1pct = sweep[sweep["total"] <= best_b + 0.5]
if len(within_1pct) > 0:
    knee_H = int(within_1pct.iloc[0]["H"])
    knee_b = within_1pct.iloc[0]["total"]
    ax.axvline(knee_H, color="purple", ls=":", lw=1.5)
    ax.text(knee_H, BASELINE_A + 1, f"Knee H={knee_H}", fontsize=9, ha="left", color="purple")

plt.tight_layout()
out_dir = ROOT / "outputs/plots"
out_dir.mkdir(exist_ok=True)
out = out_dir / "19_horizon_sensitivity.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved -> {out}")

# Print summary
print("\nHorizon sweep summary:")
print(sweep[["H", "april", "sept", "total", "vs_a", "time"]].to_string(index=False))
print(f"\nBest H = {best_H}  ->  bill = EUR {best_b:+.2f}  (vs A = EUR {best_b - BASELINE_A:+.2f})")
