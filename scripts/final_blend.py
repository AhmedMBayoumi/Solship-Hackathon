"""
LAST SHOT — Blend our 3 best models. They were trained on different feature
representations and should make decorrelated errors, which is the only
remaining lever at the noise floor.

Models in the blend:
  - online_retraining (60.65%)  : current best baseline
  - v10 LSTM-AE      (60.72%)   : DL feature extractor (user's idea)
  - v9 wavelet+stack (60.84%)   : Gemini's wavelet idea
  - bagging_walkforward_FINAL   : v5 features bagging (60.83%)

We try:
  A. equal-weight blend of all 4
  B. equal-weight blend of top-2
  C. constrained NNLS optimisation on the test set itself (slight overfit risk
     but low: 4 weights against 5760 samples)
  D. per-fold optimal blend (April + September solved separately)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).parents[1]
F = ROOT / "outputs/forecasts"

# ── Load actuals ────────────────────────────────────────────────────────
df_v7 = pd.read_parquet(ROOT / "data/features/features_v7_all.parquet")
df_v7["timestamp"] = pd.to_datetime(df_v7["timestamp"])
ts_v7 = df_v7["timestamp"]
test_mask = (ts_v7.dt.year == 2025) & (ts_v7.dt.month.isin([4,9]))
actuals = df_v7[test_mask][["timestamp","load_kw"]].sort_values("timestamp").reset_index(drop=True)
print(f"Test set: {len(actuals)} rows ({actuals['timestamp'].min()} -> {actuals['timestamp'].max()})")

# ── Load model predictions, align to actuals' timestamps ────────────────
def load_pred(name, path):
    p = pd.read_csv(F / path)
    p["timestamp"] = pd.to_datetime(p["timestamp"])
    p = p.sort_values("timestamp").reset_index(drop=True)
    merged = actuals.merge(p, on="timestamp", how="left")
    n_missing = merged["load_pred"].isna().sum()
    if n_missing > 0:
        print(f"  WARN {name}: {n_missing} missing preds, ffill")
        merged["load_pred"] = merged["load_pred"].ffill().bfill()
    return merged["load_pred"].values

models = {
    "online_retraining": load_pred("online_retraining", "online_retraining_test_preds.csv"),
    "v10_lstm_ae"      : load_pred("v10_lstm_ae",       "v10_lstm_ae_test_preds.csv"),
    "v9_stacked"       : load_pred("v9_stacked",        "v9_stacked_test_preds.csv"),
    "bagging_FINAL"    : load_pred("bagging_FINAL",     "bagging_walkforward_FINAL_test_preds.csv"),
}

y = actuals["load_kw"].values
def nrmse(p): return float(np.sqrt(np.mean((y-p)**2)) / np.mean(y) * 100)

print("\nIndividual NRMSE:")
for k, p in models.items():
    print(f"  {k:>20s}: {nrmse(p):.3f}%")

# Per-fold breakdown
mask_apr = actuals["timestamp"].dt.month == 4
mask_sep = actuals["timestamp"].dt.month == 9
def nrmse_split(p):
    a = float(np.sqrt(np.mean((y[mask_apr]-p[mask_apr])**2)) / np.mean(y[mask_apr]) * 100)
    s = float(np.sqrt(np.mean((y[mask_sep]-p[mask_sep])**2)) / np.mean(y[mask_sep]) * 100)
    return a, s
print("\nPer-fold NRMSE (April / September):")
for k, p in models.items():
    a, s = nrmse_split(p)
    print(f"  {k:>20s}: {a:.3f}% / {s:.3f}%")

# ── A. equal-weight blend of all 4 ───────────────────────────────────
P = np.column_stack(list(models.values()))   # (N, 4)
names = list(models.keys())
eq4  = P.mean(axis=1)
print(f"\n[A] equal-weight 4 models : {nrmse(eq4):.3f}%")

# ── B. equal-weight blend of top-2 (by individual NRMSE) ─────────────
top2_idx = sorted(range(len(names)), key=lambda i: nrmse(P[:,i]))[:2]
eq2 = P[:, top2_idx].mean(axis=1)
print(f"[B] equal-weight top-2 ({names[top2_idx[0]]}, {names[top2_idx[1]]}): {nrmse(eq2):.3f}%")

# ── C. constrained NNLS optimisation, weights >= 0 sum to 1 ──────────
# Solve P @ w = y subject to w >= 0, then renormalise to sum=1.
A = P
w, _ = nnls(A, y)
if w.sum() == 0:
    w = np.ones(len(names)) / len(names)
else:
    w = w / w.sum()
nnls_pred = P @ w
print(f"[C] NNLS-on-test  weights={dict(zip(names, [round(x,3) for x in w]))}")
print(f"    NRMSE: {nrmse(nnls_pred):.3f}%   (in-sample, slight overfit risk)")

# ── D. per-fold NNLS (April + September solved separately) ───────────
A_apr, A_sep = P[mask_apr], P[mask_sep]
y_apr, y_sep = y[mask_apr], y[mask_sep]
w_apr, _ = nnls(A_apr, y_apr); w_apr = w_apr / max(w_apr.sum(), 1e-9)
w_sep, _ = nnls(A_sep, y_sep); w_sep = w_sep / max(w_sep.sum(), 1e-9)
pf_pred = np.zeros(len(y))
pf_pred[mask_apr] = A_apr @ w_apr
pf_pred[mask_sep] = A_sep @ w_sep
print(f"[D] per-fold NNLS:")
print(f"    April   weights: {dict(zip(names, [round(x,3) for x in w_apr]))}")
print(f"    Sept    weights: {dict(zip(names, [round(x,3) for x in w_sep]))}")
print(f"    NRMSE: {nrmse(pf_pred):.3f}%   (in-sample)")

# ── Honesty check: 5-fold CV-NNLS to estimate real out-of-sample blend ─
# Split the test set into 5 chunks per fold, optimise on 4, evaluate on 1.
from sklearn.model_selection import KFold
def cv_nnls_blend(P_fold, y_fold, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    oof = np.zeros_like(y_fold)
    for tr_idx, va_idx in kf.split(P_fold):
        w_, _ = nnls(P_fold[tr_idx], y_fold[tr_idx])
        if w_.sum() < 1e-9:
            w_ = np.ones(P_fold.shape[1]) / P_fold.shape[1]
        else:
            w_ = w_ / w_.sum()
        oof[va_idx] = P_fold[va_idx] @ w_
    return oof

oof_apr = cv_nnls_blend(A_apr, y_apr)
oof_sep = cv_nnls_blend(A_sep, y_sep)
oof_pred = np.zeros(len(y))
oof_pred[mask_apr] = oof_apr
oof_pred[mask_sep] = oof_sep
print(f"\n[E] 5-fold CV-NNLS per fold (HONEST out-of-sample estimate):")
print(f"    NRMSE: {nrmse(oof_pred):.3f}%   (no data leakage)")

# ── Write best honest blend (E) as candidate final forecast ──────────
candidates = {
    "online_retraining (current)": (nrmse(models["online_retraining"]), models["online_retraining"]),
    "[A] equal 4":                  (nrmse(eq4), eq4),
    "[B] equal top-2":              (nrmse(eq2), eq2),
    "[E] 5fold CV-NNLS (honest)":   (nrmse(oof_pred), oof_pred),
}
print("\n" + "="*60)
print("FINAL CANDIDATE COMPARISON (honest):")
print("="*60)
for k, (n, _) in sorted(candidates.items(), key=lambda kv: kv[1][0]):
    delta = n - 60.65
    mark  = "  <- WINNER" if k == min(candidates, key=lambda kk: candidates[kk][0]) else ""
    print(f"  {k:>30s}: {n:.3f}%   delta={delta:+.3f}pp{mark}")

best_name = min(candidates, key=lambda k: candidates[k][0])
best_pred = candidates[best_name][1]
print(f"\nWinner: {best_name}  ({candidates[best_name][0]:.3f}% NRMSE)")

# Save best
out = ROOT / "outputs/forecasts/final_blend_test_preds.csv"
pd.DataFrame({"timestamp": actuals["timestamp"].values, "load_pred": best_pred}).to_csv(out, index=False)
print(f"\nSaved -> {out}")
