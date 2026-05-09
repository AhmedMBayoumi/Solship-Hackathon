"""
Extension 1: Horizon sensitivity sweep.
Tests 10 horizons from 1-step (15 min) to 672-step (1 week).
Generates savings-vs-H table and curve plot.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

HORIZONS = [1, 4, 8, 16, 24, 48, 96, 192, 288, 672]
BASELINE_A = -7.57
OUT_DIR = Path(__file__).parents[2] / "outputs"


def run_sweep(df_2025: pd.DataFrame, forecast_fn_factory, save: bool = True) -> pd.DataFrame:
    """
    Sweep all horizons and tabulate results.
    forecast_fn_factory(df_month) -> callable(t, H) -> np.ndarray
    """
    from src.controller.mpc_loop import run_both_months

    rows = []
    for H in HORIZONS:
        print(f"\n--- H = {H} ({H*15} min) ---")
        t0 = time.time()
        _, bill = run_both_months(df_2025, forecast_fn_factory, H, verbose=True, label=f"H={H}")
        elapsed = time.time() - t0
        savings = round(BASELINE_A - bill["net_bill"], 4)  # negative bill is better → savings = A - ours
        savings_pct = round(savings / abs(BASELINE_A) * 100, 2) if BASELINE_A != 0 else 0
        rows.append({
            "H":           H,
            "H_hours":     round(H * 15 / 60, 2),
            "net_bill":    bill["net_bill"],
            "savings_eur": savings,
            "savings_pct": savings_pct,
            "wall_time_s": round(elapsed, 2),
        })
        print(f"  Savings vs A: EUR {savings:+.2f} ({savings_pct:+.2f}%)  |  {elapsed:.1f}s")

    df_sweep = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("HORIZON SENSITIVITY RESULTS")
    print("=" * 70)
    print(df_sweep.to_string(index=False))
    best_row = df_sweep.loc[df_sweep["savings_eur"].idxmax()]
    print(f"\nBest H: {int(best_row['H'])} ({best_row['H_hours']}h) → EUR {best_row['savings_eur']:+.2f} savings")

    if save:
        df_sweep.to_csv(OUT_DIR / "horizon_sweep.csv", index=False)
        _plot_sweep(df_sweep)
    return df_sweep


def _plot_sweep(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_bill = "#1f77b4"
    color_time = "#ff7f0e"

    ax1.plot(df["H"], df["net_bill"], "o-", color=color_bill, lw=2, label="Net bill (EUR)")
    ax1.axhline(y=BASELINE_A, color="red", ls="--", lw=1.5, label=f"Baseline A ({BASELINE_A} EUR)")
    ax1.set_xlabel("Forecast horizon H (steps, 15-min each)", fontsize=11)
    ax1.set_ylabel("Total bill Apr+Sep 2025 (EUR)", color=color_bill, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_bill)
    ax1.set_xscale("log")
    ax1.set_xticks(df["H"])
    ax1.set_xticklabels([f"H={h}\n({h*15//60}h{h*15%60:02d}m)" for h in df["H"]], fontsize=7)

    ax2 = ax1.twinx()
    ax2.bar(df["H"], df["wall_time_s"], width=df["H"]*0.3, alpha=0.3, color=color_time, label="Wall time (s)")
    ax2.set_ylabel("Wall time (s)", color=color_time, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_time)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title("Horizon Sensitivity - Bill vs Horizon Length\nSolship Energy AI Hackathon 2026", fontsize=12)
    plt.tight_layout()

    out = OUT_DIR / "plots" / "19_horizon_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")
