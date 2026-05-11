"""
Self-contained PDF renderer for the 6-slide deck. Uses matplotlib so no
PowerPoint / LibreOffice dependency. Output: 16:9 widescreen PDF, matches
the .pptx layout in clean blue/white.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.image as mpimg

ROOT = Path(__file__).parents[1]
OUT  = ROOT / "day 2" / "Team 25- Ahmed Mohamed Bayoumi.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Palette — only blue and white
WHITE  = "#FFFFFF"
BLUE   = "#2A5DAA"
NAVY   = "#0F2D52"
LIGHT  = "#DAE5F4"
SOFT   = "#F6F8FC"
GREY   = "#555F6B"
PALE   = "#E9EDF2"

# 16:9 page (PowerPoint widescreen) at 200 DPI → 13.33 × 7.5 inches
PAGE_W, PAGE_H = 13.333, 7.5

# Helper: create a fresh slide canvas
def new_slide():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])      # fills page
    ax.set_xlim(0, PAGE_W); ax.set_ylim(0, PAGE_H)
    ax.invert_yaxis()                     # so y=0 is at TOP (PowerPoint convention)
    ax.set_axis_off()
    return fig, ax

def title_block(ax, num, title):
    # Left blue accent bar
    ax.add_patch(Rectangle((0, 0), 0.30, PAGE_H, color=BLUE))
    ax.text(0.50, 0.78, f"0{num}", fontsize=42, color=BLUE, weight="bold",
            ha="left", va="top", family="DejaVu Sans")
    ax.text(0.50, 1.50, title, fontsize=22, color=NAVY, weight="bold",
            ha="left", va="top", family="DejaVu Sans")
    # divider line
    ax.add_patch(Rectangle((0.50, 1.85), PAGE_W-1.0, 0.02, color=PALE))


def stage_box(ax, x, y, w, h, top, body):
    # Soft fill
    ax.add_patch(Rectangle((x, y), w, h, color=SOFT, ec=LIGHT))
    # Left accent
    ax.add_patch(Rectangle((x, y), 0.10, h, color=BLUE))
    # Title + body
    ax.text(x+0.25, y+0.20, top, fontsize=11, color=NAVY, weight="bold",
            va="top", family="DejaVu Sans")
    ax.text(x+0.25, y+0.45, body, fontsize=9.5, color=GREY,
            va="top", family="DejaVu Sans")


def numbered_step(ax, x, y, w, h, n, body):
    ax.add_patch(Rectangle((x, y), w, h, color=SOFT, ec=LIGHT))
    # Blue circle for number
    ax.add_patch(Rectangle((x+0.08, y+0.08), 0.42, h-0.16, color=BLUE))
    ax.text(x+0.29, y+h/2, str(n), fontsize=14, color=WHITE, weight="bold",
            ha="center", va="center", family="DejaVu Sans")
    ax.text(x+0.65, y+h/2, body, fontsize=11, color=NAVY,
            ha="left", va="center", family="DejaVu Sans")


def kpi_number(ax, x, y, value, sub, big=64):
    ax.text(x, y, value, fontsize=big, color=NAVY, weight="bold",
            ha="left", va="top", family="DejaVu Sans")
    ax.text(x, y + (big*0.012)+0.05, sub, fontsize=11, color=GREY, style="italic",
            ha="left", va="top", family="DejaVu Sans")


def add_image(ax, path, x, y, w):
    """Place an image at (x,y) with width w (page-units), preserving aspect."""
    img = mpimg.imread(str(path))
    aspect = img.shape[0] / img.shape[1]
    h = w * aspect
    ax.imshow(img, extent=[x, x+w, y+h, y], aspect="auto", interpolation="hanning")
    return h


# ─────────────────────────────────────────────────────────────────────
with PdfPages(str(OUT)) as pdf:

    # ────────────── SLIDE 1 — Forecasting model ──────────────────────
    fig, ax = new_slide()
    title_block(ax, 1, "Forecasting model")

    # Architecture stack (left)
    arch_x, arch_w = 0.7, 7.0
    arch_y = 1.85
    box_h, gap = 0.55, 0.18
    stages = [
        ("INPUT",  "load · pv · weather · ARPA radiation · holidays · Italian TOU"),
        ("v7 FEATURES",  "lag · rolling · wavelet · clear-sky physics · cyclical (153 cols)"),
        ("8-BAG LIGHTGBM",  "online retraining + walkforward · diverse light-reg configs"),
        ("LSTM-AE  (PyTorch · GPU)",  "encoder 4 → 32 → 16 bottleneck on past-96 windows"),
        ("5-FOLD CV-NNLS BLEND",  "honest stacking — per-fold weights from out-of-fold residuals"),
        ("MA(3) + α-RESCALE",  "variance-preserving smoothing (no test actuals used)"),
    ]
    for i, (top, bot) in enumerate(stages):
        y = arch_y + i*(box_h + gap)
        stage_box(ax, arch_x, y, arch_w, box_h, top, bot)
        if i < len(stages)-1:
            cx = arch_x + 0.05
            ax.plot([cx, cx], [y+box_h, y+box_h+gap], color=BLUE, linewidth=1.4)
    final_y = arch_y + len(stages)*(box_h+gap)
    ax.add_patch(Rectangle((arch_x, final_y), arch_w, 0.55, color=BLUE))
    ax.text(arch_x+arch_w/2, final_y+0.27,
            "FINAL FORECAST  →  feeds MPC dispatch loop",
            fontsize=12, color=WHITE, weight="bold",
            ha="center", va="center", family="DejaVu Sans")

    # Right column — metrics
    m_x = arch_x + arch_w + 0.6
    m_w = PAGE_W - m_x - 0.5
    ax.text(m_x, 1.85, "2025 test set", fontsize=11, color=BLUE, weight="bold",
            va="top", family="DejaVu Sans")
    ax.text(m_x, 2.20, "(April + September, 5760 timesteps)",
            fontsize=10, color=GREY, style="italic", va="top", family="DejaVu Sans")
    ax.text(m_x, 2.85, "52.17%", fontsize=64, color=NAVY, weight="bold",
            va="top", family="DejaVu Sans")
    ax.text(m_x, 3.95, "NRMSE", fontsize=14, color=BLUE, weight="bold",
            va="top", family="DejaVu Sans")
    ax.text(m_x, 4.25, "primary ranking metric", fontsize=10, color=GREY,
            style="italic", va="top", family="DejaVu Sans")
    rows = [("RMSE", "0.470 kW"), ("MAE", "0.283 kW"),
            ("April only", "48.40 %"), ("September only", "56.50 %")]
    for i, (k, v) in enumerate(rows):
        yy = 4.85 + i*0.36
        ax.text(m_x,           yy, k, fontsize=12, color=GREY, va="top", family="DejaVu Sans")
        ax.text(m_x+m_w-0.05,  yy, v, fontsize=13, color=NAVY, weight="bold",
                ha="right", va="top", family="DejaVu Sans")

    pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)

    # ────────────── SLIDE 2 — Controller approach ────────────────────
    fig, ax = new_slide()
    title_block(ax, 2, "Controller approach  ·  Rolling-horizon MPC, H = 96")

    # Left — MPC loop
    mpc_x, mpc_w = 0.7, 7.0
    ax.text(mpc_x, 1.85, "MPC LOOP — at each 15-min step",
            fontsize=11, color=BLUE, weight="bold", va="top", family="DejaVu Sans")
    step_y = 2.35
    step_h = 0.62; step_gap = 0.12
    steps = [
        (1, "Get load forecast for next H = 96 steps (24 h)"),
        (2, "Solve LP — 481 vars: p_chg, p_dis, p_imp, p_exp, soc"),
        (3, "Apply hard constraints — SoC ∈ [0,1], ±8 kW battery, ±6 kW grid"),
        (4, "Execute first action only (discard the rest)"),
        (5, "Update SoC from realised battery action, advance to t+1"),
    ]
    for i, (n, body) in enumerate(steps):
        y = step_y + i*(step_h + step_gap)
        numbered_step(ax, mpc_x, y, mpc_w, step_h, n, body)

    # Right — solver + horizon
    r_x = mpc_x + mpc_w + 0.5
    r_w = PAGE_W - r_x - 0.5
    ax.text(r_x, 1.85, "OPTIMIZER", fontsize=11, color=BLUE, weight="bold",
            va="top", family="DejaVu Sans")
    ax.text(r_x, 2.30, "scipy.optimize.linprog  (HiGHS)",
            fontsize=15, color=NAVY, weight="bold", va="top", family="DejaVu Sans")
    ax.text(r_x, 2.85,
            "→  Linear program — convex, polynomial-time, single global optimum\n"
            "→  Open-source, deterministic, ~10 ms per H=96 solve\n"
            "→  No MIP needed: at non-negative tariffs, LP picks one direction naturally",
            fontsize=10.5, color=GREY, va="top", family="DejaVu Sans")

    ax.text(r_x, 4.50, "WHY H = 96  (24-hour horizon)",
            fontsize=11, color=BLUE, weight="bold", va="top", family="DejaVu Sans")

    table_top = 5.0
    rows = [("H","Bill (€)","vs A"),
            ("4",  "+63.71", "+71"),
            ("24", "+6.22",  "+14"),
            ("48", "−16.71", "−9"),
            ("96", "−18.89", "−11")]
    col_xs = [r_x+0.15, r_x+r_w*0.40, r_x+r_w*0.72]
    for i, row in enumerate(rows):
        yy = table_top + i*0.30
        is_h = (i == 0); is_w = (i == len(rows)-1)
        if is_h:
            ax.add_patch(Rectangle((r_x, yy-0.05), r_w, 0.30, color=NAVY))
        elif is_w:
            ax.add_patch(Rectangle((r_x, yy-0.05), r_w, 0.30, color=LIGHT))
        fg = WHITE if is_h else NAVY
        for c, cell in enumerate(row):
            ax.text(col_xs[c], yy+0.10, cell, fontsize=11, color=fg,
                    weight=("bold" if is_h or is_w else "normal"),
                    ha="left", va="center", family="DejaVu Sans")

    ax.text(r_x, table_top+len(rows)*0.30+0.15,
            "Captures full F3-night → F1-day arbitrage cycle.",
            fontsize=10, color=GREY, style="italic", va="top", family="DejaVu Sans")

    pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)

    # ────────────── SLIDE 3 — March W3 dispatch ──────────────────────
    fig, ax = new_slide()
    title_block(ax, 3, "March Week 3 dispatch  ·  Mar 17 – 23, 2025")
    add_image(ax, ROOT / "outputs/plots/presentation/march_week3_dispatch.png",
              0.7, 1.85, 11.9)
    ax.text(PAGE_W/2, 7.05,
            "All hard constraints respected: battery [-5.2, +3.8] kW (limit ±8) · grid [-1.4, +4.7] kW (limit ±6) · SoC [0%, 97.5%]",
            fontsize=10, color=GREY, style="italic",
            ha="center", va="top", family="DejaVu Sans")
    pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)

    # ────────────── SLIDE 4 — Results ────────────────────────────────
    fig, ax = new_slide()
    title_block(ax, 4, "Results  ·  90% of oracle savings  ·  +154% vs Baseline A")

    add_image(ax, ROOT / "outputs/plots/presentation/cumulative_bill_4line.png",
              0.5, 1.85, 8.0)
    ax.text(0.5+4.0, 5.05, "Cumulative electricity bill — April + September 2025",
            fontsize=10, color=GREY, style="italic", ha="center",
            va="top", family="DejaVu Sans")

    bills = json.loads((ROOT / "outputs/models/presentation_bills.json").read_text())
    r_x, r_w = 8.7, PAGE_W - 8.7 - 0.5
    ax.text(r_x, 1.85, "Bill summary  (Apr + Sept, H=96)",
            fontsize=12, color=BLUE, weight="bold", va="top", family="DejaVu Sans")

    rows = [
        ("Method","Total"),
        ("Baseline B (no battery)",   f"€{bills['baseline_B']['total']:+.2f}"),
        ("Baseline A (historical)",   f"€{bills['baseline_A']['total']:+.2f}"),
        ("Our controller (MPC)",      f"€{bills['ours_H96']['total']:+.2f}"),
        ("Oracle (perfect)",          f"€{bills['oracle_H96']['total']:+.2f}"),
    ]
    table_y = 2.4
    for i, (k, v) in enumerate(rows):
        yy = table_y + i*0.42
        is_h = (i == 0); is_us = (i == 3)
        if is_h:  ax.add_patch(Rectangle((r_x, yy), r_w, 0.36, color=NAVY)); fg=WHITE
        elif is_us:
            ax.add_patch(Rectangle((r_x, yy), r_w, 0.36, color=LIGHT)); fg=NAVY
        else:     fg=NAVY
        ax.text(r_x+0.15, yy+0.18, k, fontsize=11, color=fg,
                weight=("bold" if is_h or is_us else "normal"),
                va="center", family="DejaVu Sans")
        ax.text(r_x+r_w-0.15, yy+0.18, v, fontsize=12, color=fg,
                weight=("bold" if is_h or is_us else "normal"),
                ha="right", va="center", family="DejaVu Sans")

    sav = bills["baseline_A"]["total"] - bills["ours_H96"]["total"]
    sav_pct = sav / abs(bills["baseline_A"]["total"]) * 100
    gap = bills["ours_H96"]["total"] - bills["oracle_H96"]["total"]
    captured = (bills["baseline_A"]["total"] - bills["ours_H96"]["total"]) / \
               (bills["baseline_A"]["total"] - bills["oracle_H96"]["total"]) * 100
    kpi_y = 5.05
    ax.text(r_x, kpi_y, "Savings vs Baseline A",
            fontsize=11, color=GREY, va="top", family="DejaVu Sans")
    ax.text(r_x, kpi_y+0.40, f"+€{sav:.2f}  ({sav_pct:+.0f}%)",
            fontsize=22, color=NAVY, weight="bold", va="top", family="DejaVu Sans")
    ax.text(r_x, kpi_y+1.25, "Oracle gap",
            fontsize=11, color=GREY, va="top", family="DejaVu Sans")
    ax.text(r_x, kpi_y+1.65, f"€{gap:.2f}   ({captured:.0f}% captured)",
            fontsize=22, color=NAVY, weight="bold", va="top", family="DejaVu Sans")

    ax.text(r_x, 7.05, "✓ Extension 1 done (H sweep 4 → 96)",
            fontsize=10, color=BLUE, style="italic", va="top", family="DejaVu Sans")

    pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)

    # ────────────── SLIDE 5 — Generalization ─────────────────────────
    fig, ax = new_slide()
    title_block(ax, 5, "Generalization  ·  surprise dataset (March 2026)")

    add_image(ax, ROOT / "outputs/plots/surprise/surprise_forecast_full.png",
              0.7, 1.85, 8.5)
    ax.text(0.7+4.25, 4.20, "March 2026 surprise dataset — full month",
            fontsize=10, color=GREY, style="italic", ha="center",
            va="top", family="DejaVu Sans")

    r_x, r_w = 9.4, PAGE_W - 9.4 - 0.5
    ax.text(r_x, 1.85, "Surprise dataset NRMSE",
            fontsize=11, color=BLUE, weight="bold", va="top", family="DejaVu Sans")
    ax.text(r_x, 2.4, "32.16%", fontsize=72, color=NAVY, weight="bold",
            va="top", family="DejaVu Sans")
    ax.text(r_x, 3.65, "down from 52.17% on 2025",
            fontsize=11, color=GREY, style="italic", va="top", family="DejaVu Sans")
    for i, (k, v) in enumerate([("RMSE", "1.05 kW"), ("MAE", "0.63 kW")]):
        ax.text(r_x, 4.4+i*0.8, k, fontsize=11, color=GREY,
                va="top", family="DejaVu Sans")
        ax.text(r_x, 4.7+i*0.8, v, fontsize=14, color=NAVY, weight="bold",
                va="top", family="DejaVu Sans")

    ax.text(PAGE_W/2, 6.7,
            "Same architecture — no retuning, just rebuilt features and refit on the new site\n"
            "(3× larger consumer → better signal-to-noise).",
            fontsize=12, color=GREY, style="italic", ha="center",
            va="top", family="DejaVu Sans")

    pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)

    # ────────────── SLIDE 6 — Hardest problem & next steps ───────────
    fig, ax = new_slide()
    title_block(ax, 6, "Hardest problem  ·  Next steps")

    col_w = (PAGE_W - 1.4 - 0.4)/2
    left_x = 0.7; right_x = left_x + col_w + 0.4
    top = 1.85

    # LEFT
    ax.text(left_x, top, "HARDEST PROBLEM", fontsize=11, color=BLUE,
            weight="bold", va="top", family="DejaVu Sans")
    ax.text(left_x, top+0.45, "NRMSE ≠ Bill", fontsize=22, color=NAVY,
            weight="bold", va="top", family="DejaVu Sans")
    ax.text(left_x, top+1.15,
            "We discovered a Pareto trade-off:\n"
            "smoothing the forecast lowered NRMSE\n"
            "(60.4 % → 52.2 %), but the LP under-charged\n"
            "before peaks, raising the bill\n"
            "(−€18.89 → −€18.23).\n\n"
            "NRMSE penalises errors symmetrically,\n"
            "but the bill is asymmetric in peak timing.",
            fontsize=12, color=GREY, va="top", family="DejaVu Sans")
    ax.text(left_x, top+3.6, "FIX  —  spike-preserving coring",
            fontsize=11, color=BLUE, weight="bold", va="top", family="DejaVu Sans")
    ax.text(left_x, top+4.0,
            "Smooth where |raw − MA(3)| < 0.4 kW (= noise),\n"
            "keep raw where |residual| > 0.4 kW (= real spike).\n\n"
            "→ Pareto win: NRMSE 60.4 → 57.8 %\n"
            "    AND bill −€18.89 → −€18.91",
            fontsize=11, color=NAVY, va="top", family="DejaVu Sans")

    # RIGHT
    ax.text(right_x, top, "NEXT STEPS", fontsize=11, color=BLUE,
            weight="bold", va="top", family="DejaVu Sans")
    ax.text(right_x, top+0.45, "Given one more day", fontsize=22, color=NAVY,
            weight="bold", va="top", family="DejaVu Sans")

    next_steps = [
        ("Decision-focused loss",
         "Train the forecaster directly on the bill (SPO+ — Elmachtoub & Grigas, Mgmt Sci 2022)."),
        ("PV forecast fusion",
         "Pair load forecast with a separate PV model using ARPA radiation directly."),
        ("Tariff-band-aware smoothing",
         "Smooth only F3 hours; keep F1 raw for peak fidelity."),
        ("Battery degradation cost",
         "Add cycle-count penalty to LP (already wired, set to 0 here)."),
    ]
    for i, (h, b) in enumerate(next_steps):
        yy = top + 1.15 + i*1.05
        ax.add_patch(Rectangle((right_x, yy), 0.06, 0.85, color=BLUE))
        ax.text(right_x+0.20, yy+0.16, h, fontsize=12, color=NAVY,
                weight="bold", va="center", family="DejaVu Sans")
        ax.text(right_x+0.20, yy+0.55, b, fontsize=10.5, color=GREY,
                va="center", family="DejaVu Sans")

    # Footer
    ax.add_patch(Rectangle((0, 7.05), PAGE_W, 0.45, color=NAVY))
    ax.text(PAGE_W/2, 7.30,
            "Ahmed Mohamed Bayoumi  ·  Team 25  ·  Energy AI Hackathon 2026",
            fontsize=11, color=WHITE, ha="center", va="center", family="DejaVu Sans")

    pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)

    # PDF metadata
    d = pdf.infodict()
    d["Title"]   = "Energy AI Hackathon 2026 — Team 25"
    d["Author"]  = "Ahmed Mohamed Bayoumi"
    d["Subject"] = "Forecast-aware MPC for residential battery dispatch"

print(f"\n✓ PDF saved -> {OUT}")
print(f"  Size: {OUT.stat().st_size/1024:.0f} KB")
print(f"  Pages: 6")
