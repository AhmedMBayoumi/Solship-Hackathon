"""Draw the forecasting model architecture as a clean infographic
(LSTM-AE neurons → 8-bag LightGBM trees → final blend layer → output).
Saves PNG that gets embedded into slide 1 of the presentation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).parents[1]
OUT  = ROOT / "outputs/plots/presentation/architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Palette
BLUE   = "#2A5DAA"
NAVY   = "#0F2D52"
LIGHT  = "#DAE5F4"
SOFT   = "#F6F8FC"
GREY   = "#555F6B"
WHITE  = "#FFFFFF"
PALE   = "#E9EDF2"

fig, ax = plt.subplots(figsize=(13.5, 6.0), facecolor=WHITE)
ax.set_xlim(0, 27); ax.set_ylim(0, 12)
ax.set_aspect("equal"); ax.set_axis_off()


# ─── Section 1: INPUT (time-series windows) ───────────────────────────
sx = 0.5; sw = 2.8
# Title
ax.text(sx + sw/2, 11.4, "INPUT", fontsize=11, color=BLUE, weight="bold",
        ha="center", va="top", family="DejaVu Sans")
ax.text(sx + sw/2, 10.85, "past 96-step windows", fontsize=9, color=GREY,
        ha="center", va="top", family="DejaVu Sans", style="italic")

# Mini stacked time-series for 4 channels (load, pv, temp, rad)
labels = ["load", "pv", "temp", "rad"]
sig_y = [9.0, 7.5, 6.0, 4.5]
np.random.seed(1)
for i, (lab, yc) in enumerate(zip(labels, sig_y)):
    # mini sparkline
    n = 20
    xs = np.linspace(sx+0.15, sx+sw-0.15, n)
    if lab == "load":
        ys = yc + 0.55 * np.abs(np.sin(np.linspace(0, 4*np.pi, n))) + 0.2*np.random.randn(n)*0.3
    elif lab == "pv":
        ys = yc + 0.55 * np.maximum(0, np.sin(np.linspace(0.5, 3.5, n))) + 0.05*np.random.randn(n)
    elif lab == "temp":
        ys = yc + 0.4 * np.sin(np.linspace(0, 2.2*np.pi, n)) + 0.05*np.random.randn(n)
    else:  # rad
        ys = yc + 0.55 * np.maximum(0, np.sin(np.linspace(0.3, 3.0, n))) + 0.05*np.random.randn(n)
    ax.plot(xs, ys, color=BLUE, lw=1.2)
    ax.text(sx-0.05, yc+0.25, lab, fontsize=8.5, color=NAVY, weight="bold",
            ha="right", va="center", family="DejaVu Sans")

# Box around input
ax.add_patch(FancyBboxPatch((sx, 4.1), sw, 6.5,
                             boxstyle="round,pad=0.05,rounding_size=0.10",
                             ec=LIGHT, fc=SOFT, lw=1.0))


# ─── Section 2: LSTM-AE encoder (drawn as a neural net) ──────────────
nx = 4.2; nw = 5.2
ax.text(nx + nw/2, 11.4, "LSTM-AE  (PyTorch · GPU)", fontsize=11, color=BLUE,
        weight="bold", ha="center", va="top", family="DejaVu Sans")
ax.text(nx + nw/2, 10.85, "encoder: 4 → 32 → 16-dim bottleneck",
        fontsize=9, color=GREY, ha="center", va="top",
        family="DejaVu Sans", style="italic")

# Three layers of neurons
layer_xs = [nx + 0.6, nx + nw/2, nx + nw - 0.6]
layer_sizes = [4, 8, 6]    # input chan, hidden representation, bottleneck
layer_centres_y = 7.3
layer_spacing = 0.85
neuron_r = 0.30

# compute y positions for each layer's neurons (centred vertically)
layer_pos = []
for i, (xc, n) in enumerate(zip(layer_xs, layer_sizes)):
    h = (n-1) * layer_spacing
    ys = [layer_centres_y - h/2 + j*layer_spacing for j in range(n)]
    layer_pos.append((xc, ys))

# Draw connections (light blue)
for i in range(len(layer_pos)-1):
    xc1, ys1 = layer_pos[i]
    xc2, ys2 = layer_pos[i+1]
    for y1 in ys1:
        for y2 in ys2:
            ax.plot([xc1, xc2], [y1, y2], color=LIGHT, lw=0.6, zorder=1)

# Draw neurons
for i, (xc, ys) in enumerate(layer_pos):
    color = BLUE if i == len(layer_pos)-1 else NAVY
    for y in ys:
        c = Circle((xc, y), neuron_r, fc=WHITE, ec=color, lw=1.5, zorder=2)
        ax.add_patch(c)
    # layer label
    label_map = {0: "input\n(4 chan)", 1: "LSTM\n(32 hid)", 2: "bottleneck\n(16-dim)"}
    ax.text(xc, layer_centres_y - max(layer_sizes)*layer_spacing/2 - 1.0,
            label_map[i], fontsize=8, color=NAVY, ha="center", va="top",
            family="DejaVu Sans")

# Box around LSTM-AE
ax.add_patch(FancyBboxPatch((nx, 4.1), nw, 6.5,
                             boxstyle="round,pad=0.05,rounding_size=0.10",
                             ec=LIGHT, fc=SOFT, lw=1.0))


# ─── Section 3: 8-bag LightGBM (drawn as small trees) ────────────────
tx = 9.8; tw = 7.4
ax.text(tx + tw/2, 11.4, "8-BAG LIGHTGBM", fontsize=11, color=BLUE,
        weight="bold", ha="center", va="top", family="DejaVu Sans")
ax.text(tx + tw/2, 10.85, "v7 features (153) + AE bottleneck (16)",
        fontsize=9, color=GREY, ha="center", va="top",
        family="DejaVu Sans", style="italic")

def draw_tree(ax, cx, cy, w, h, color):
    """Draw a stylised decision tree icon."""
    # Root
    rx, ry = cx, cy + h/2
    # Children
    c1x, c1y = cx - w*0.22, cy
    c2x, c2y = cx + w*0.22, cy
    # Leaves
    l1x, l1y = cx - w*0.36, cy - h/2
    l2x, l2y = cx - w*0.10, cy - h/2
    l3x, l3y = cx + w*0.10, cy - h/2
    l4x, l4y = cx + w*0.36, cy - h/2
    # Edges
    for (a, b) in [((rx,ry),(c1x,c1y)), ((rx,ry),(c2x,c2y)),
                   ((c1x,c1y),(l1x,l1y)), ((c1x,c1y),(l2x,l2y)),
                   ((c2x,c2y),(l3x,l3y)), ((c2x,c2y),(l4x,l4y))]:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.0, zorder=1)
    # Nodes
    for (px, py, is_leaf) in [(rx,ry,False), (c1x,c1y,False), (c2x,c2y,False),
                               (l1x,l1y,True), (l2x,l2y,True), (l3x,l3y,True), (l4x,l4y,True)]:
        if is_leaf:
            sq = Rectangle((px-0.06, py-0.06), 0.12, 0.12, fc=color, ec=color, zorder=2)
            ax.add_patch(sq)
        else:
            cir = Circle((px, py), 0.09, fc=WHITE, ec=color, lw=1.2, zorder=2)
            ax.add_patch(cir)

# 8 trees in 2 rows × 4 cols
tree_rows = 2
tree_cols = 4
tree_w = (tw - 0.8) / tree_cols
tree_h = (10.0 - 4.5) / tree_rows
for r in range(tree_rows):
    for c in range(tree_cols):
        cx = tx + 0.4 + tree_w*c + tree_w/2
        cy = 9.2 - tree_h*r - tree_h/2 + 0.3
        draw_tree(ax, cx, cy, tree_w*0.55, tree_h*0.55, BLUE)

# Bag labels under each row
for r in range(tree_rows):
    ax.text(tx+tw/2, 9.4 - tree_h*r - tree_h*0.95,
            f"bags {r*4+1}–{r*4+4}",
            fontsize=8, color=GREY, ha="center", va="top",
            family="DejaVu Sans", style="italic")

# Box around trees
ax.add_patch(FancyBboxPatch((tx, 4.1), tw, 6.5,
                             boxstyle="round,pad=0.05,rounding_size=0.10",
                             ec=LIGHT, fc=SOFT, lw=1.0))


# ─── Section 4: Final blend / DL output layer ─────────────────────────
fx = 17.8; fw = 4.8
ax.text(fx + fw/2, 11.4, "FINAL BLEND LAYER", fontsize=11, color=BLUE,
        weight="bold", ha="center", va="top", family="DejaVu Sans")
ax.text(fx + fw/2, 10.85, "5-fold CV-NNLS  +  smoothing",
        fontsize=9, color=GREY, ha="center", va="top",
        family="DejaVu Sans", style="italic")

# Draw a small fully-connected layer: inputs (8 bag preds) -> output node
in_xs = fx + 0.7
out_xs = fx + fw - 0.9
in_ys = np.linspace(8.6, 5.0, 8)
out_y = 6.8

# Connections (faint blue, thicker for nonzero NNLS weights)
nnls_w = [0.4, 0.6, 0, 0, 0, 0, 0, 0]   # illustrative: only top bags get weight
for y, w in zip(in_ys, nnls_w):
    lw = 0.6 + 2.5*w
    color = BLUE if w > 0 else LIGHT
    ax.plot([in_xs, out_xs], [y, out_y], color=color, lw=lw, zorder=1)

# Input nodes
for y in in_ys:
    ax.add_patch(Circle((in_xs, y), 0.18, fc=WHITE, ec=BLUE, lw=1.2, zorder=2))
ax.text(in_xs-0.45, np.mean(in_ys)+0.2, "8 bag preds",
        fontsize=8, color=NAVY, rotation=90, ha="center", va="center",
        family="DejaVu Sans")

# Output node (bigger, navy)
ax.add_patch(Circle((out_xs, out_y), 0.35, fc=NAVY, ec=NAVY, zorder=3))
ax.text(out_xs, out_y, "ŷ", fontsize=14, color=WHITE, weight="bold",
        ha="center", va="center", family="DejaVu Sans")

# Smoothing badge below
ax.add_patch(FancyBboxPatch((fx + 0.35, 4.4), fw - 0.7, 0.55,
                             boxstyle="round,pad=0.03,rounding_size=0.10",
                             ec=BLUE, fc=BLUE, lw=0))
ax.text(fx + fw/2, 4.68, "MA(3) + α-rescale", fontsize=10, color=WHITE,
        weight="bold", ha="center", va="center", family="DejaVu Sans")

# Box around final layer
ax.add_patch(FancyBboxPatch((fx, 4.1), fw, 6.5,
                             boxstyle="round,pad=0.05,rounding_size=0.10",
                             ec=LIGHT, fc=SOFT, lw=1.0))


# ─── Arrows between sections ───────────────────────────────────────
def section_arrow(x0, x1, y, color=BLUE):
    a = FancyArrowPatch((x0, y), (x1, y), color=color,
                         arrowstyle="-|>", mutation_scale=20, lw=2.0, zorder=4)
    ax.add_patch(a)

section_arrow(sx + sw + 0.05,    nx - 0.05,         7.3)
section_arrow(nx + nw + 0.05,    tx - 0.05,         7.3)
section_arrow(tx + tw + 0.05,    fx - 0.05,         7.3)


# ─── Final output box ──────────────────────────────────────────────
out_x_box = fx + 0.4
out_y_box = 2.7
out_w_box = fw - 0.8
out_h_box = 0.85
ax.add_patch(FancyBboxPatch((out_x_box, out_y_box), out_w_box, out_h_box,
                             boxstyle="round,pad=0.04,rounding_size=0.10",
                             ec=NAVY, fc=NAVY, lw=0))
ax.text(out_x_box + out_w_box/2, out_y_box + out_h_box/2,
        "FORECAST  ŷ(t)", fontsize=14, color=WHITE, weight="bold",
        ha="center", va="center", family="DejaVu Sans")

# arrow from blend → output
arr = FancyArrowPatch((fx + fw/2, 4.1),
                      (out_x_box + out_w_box/2, out_y_box + out_h_box + 0.05),
                      color=NAVY, arrowstyle="-|>", mutation_scale=20, lw=2.5, zorder=4)
ax.add_patch(arr)

# Bottom caption
ax.text(13.5, 1.5,
        "Two-stream input → encoder bottleneck + tabular features → 8 bagged trees → "
        "weighted blend → smoothing → forecast",
        fontsize=10, color=GREY, ha="center", va="center",
        style="italic", family="DejaVu Sans")

fig.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print(f"Saved -> {OUT}")
print(f"Size: {OUT.stat().st_size/1024:.0f} KB")
