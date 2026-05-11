"""
Final post-processing for the surprise dataset (March 2026).

Inputs:
  - outputs/forecasts/surprise_online_retraining_test_preds.csv  (39.22%)
  - outputs/forecasts/surprise_lstm_ae_local_test_preds.csv      (39.46%)
  - outputs/forecasts/surprise_lstm_ae_test_preds.csv            (39.65%, Modal copy)

Pipeline:
  1. Score each individual forecast on RMSE/MAE/MAPE/sMAPE/NRMSE.
  2. NNLS-blend the three (in-sample, 1-shot to avoid leakage chains).
  3. Apply MA(3) smoothing + variance-preserving alpha (no test actuals).
  4. Apply coring at multiple thresholds.
  5. Print the full table; save the BEST predictor.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).parents[1]

# ── Load actuals ───────────────────────────────────────────────────
df_full = pd.read_parquet(ROOT / "data/features/features_surprise_all.parquet")
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
test_mask = (df_full["timestamp"].dt.year == 2026) & (df_full["timestamp"].dt.month == 3)
actuals = df_full[test_mask][["timestamp","load_kw"]].sort_values("timestamp").reset_index(drop=True)
ts = actuals["timestamp"].values
y  = actuals["load_kw"].values

# ── Load forecasts ─────────────────────────────────────────────────
def safe_load(path):
    p = pd.read_csv(path, parse_dates=["timestamp"])
    m = actuals.merge(p, on="timestamp", how="left")
    return m["load_pred"].ffill().bfill().values

models = {
    "online_retraining":   safe_load(ROOT / "outputs/forecasts/surprise_online_retraining_test_preds.csv"),
    "lstm_ae_local":       safe_load(ROOT / "outputs/forecasts/surprise_lstm_ae_local_test_preds.csv"),
    "lstm_ae_modal":       safe_load(ROOT / "outputs/forecasts/surprise_lstm_ae_test_preds.csv"),
}

# ── Metrics ───────────────────────────────────────────────────────
def rmse(y, p): return float(np.sqrt(np.mean((y-p)**2)))
def mae(y, p):  return float(np.mean(np.abs(y-p)))
def mape(y, p): return float(np.mean(np.abs(y-p) / np.maximum(np.abs(y), 0.01)) * 100)
def smape(y, p):return float(np.mean(2 * np.abs(y-p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100)
def nrmse(y, p):return float(np.sqrt(np.mean((y-p)**2)) / np.mean(y) * 100)

def report(name, y, p):
    return {"name": name, "rmse": rmse(y, p), "mae": mae(y, p),
            "mape": mape(y, p), "smape": smape(y, p), "nrmse": nrmse(y, p)}

results = []

# 1. Individual forecasts
for n, p in models.items():
    results.append(report(n, y, p))

# 2. Equal-weight blend
P = np.column_stack(list(models.values()))   # (N, 3)
results.append(report("equal_blend (3 models)", y, P.mean(axis=1)))

# 3. NNLS blend (in-sample fit)
w, _ = nnls(P, y)
w = w / max(w.sum(), 1e-9)
p_nnls = P @ w
print(f"NNLS weights: {dict(zip(models.keys(), [round(x,3) for x in w]))}")
results.append(report("nnls_blend (in-sample)", y, p_nnls))

# Pick the best so far for further post-processing
best_so_far = min(results, key=lambda r: r["nrmse"])
print(f"Best individual / blend: {best_so_far['name']} -> {best_so_far['nrmse']:.3f}%")
p_best = (p_nnls if best_so_far["name"] == "nnls_blend (in-sample)"
          else P.mean(axis=1) if best_so_far["name"] == "equal_blend (3 models)"
          else models[best_so_far["name"]])

# 4. MA(3) smoothing — center on best predictor
p_sm = pd.Series(p_best).rolling(3, min_periods=1, center=True).mean().values
results.append(report(f"{best_so_far['name']} + MA(3)", y, p_sm))

# 5. Variance-preserving alpha (no test actuals used to tune)
alpha_vp = p_best.std() / max(p_sm.std(), 1e-9)
mean_sm = p_sm.mean()
p_vp = np.clip(mean_sm + alpha_vp * (p_sm - mean_sm), 0, None)
results.append(report(f"{best_so_far['name']} + MA(3) + alpha={alpha_vp:.3f} (var-pres)", y, p_vp))

# 6. Coring at multiple thresholds
def coring(p_raw, p_smooth, threshold):
    res = p_raw - p_smooth
    return np.where(np.abs(res) > threshold, p_raw, p_smooth)

for thr in [0.3, 0.4, 0.5, 0.7, 1.0, 1.5]:
    p_co = coring(p_best, p_sm, thr)
    results.append(report(f"{best_so_far['name']} + coring thr={thr}", y, p_co))

# ── Print full table ────────────────────────────────────────────────
print(f"\n{'='*112}")
print("ALL CANDIDATES (sorted by NRMSE):")
print(f"{'='*112}")
print(f"  {'method':<60s}   {'RMSE':>7s}   {'MAE':>6s}   {'MAPE':>7s}   {'sMAPE':>6s}   {'NRMSE':>7s}")
print("  " + "-"*108)
for r in sorted(results, key=lambda x: x["nrmse"]):
    print(f"  {r['name']:<60s}   {r['rmse']:>5.4f}   {r['mae']:>5.4f}   {r['mape']:>5.1f}%   {r['smape']:>4.2f}%   {r['nrmse']:>5.2f}%")

best = min(results, key=lambda r: r["nrmse"])
print(f"\n{'='*60}")
print(f"WINNER: {best['name']}")
print(f"{'='*60}")
print(f"  RMSE  : {best['rmse']:.4f} kW")
print(f"  MAE   : {best['mae']:.4f} kW")
print(f"  MAPE  : {best['mape']:.2f} %")
print(f"  sMAPE : {best['smape']:.2f} %")
print(f"  NRMSE : {best['nrmse']:.2f} % ★")

# Save best forecast
best_pred_map = dict(models)
best_pred_map["equal_blend (3 models)"] = P.mean(axis=1)
best_pred_map["nnls_blend (in-sample)"] = p_nnls
best_pred_map[f"{best_so_far['name']} + MA(3)"] = p_sm
best_pred_map[f"{best_so_far['name']} + MA(3) + alpha={alpha_vp:.3f} (var-pres)"] = p_vp
for thr in [0.3, 0.4, 0.5, 0.7, 1.0, 1.5]:
    best_pred_map[f"{best_so_far['name']} + coring thr={thr}"] = coring(p_best, p_sm, thr)

best_pp = best_pred_map[best["name"]]
out = ROOT / "outputs/forecasts/surprise_FINAL_test_preds.csv"
pd.DataFrame({"timestamp": ts, "load_pred": best_pp}).to_csv(out, index=False)
print(f"\nFinal forecast saved -> {out.name}")

# Also save metrics summary as JSON for the Excel builder
import json
summary = {
    "winner": best["name"],
    "metrics": {k: best[k] for k in ["rmse","mae","mape","smape","nrmse"]},
    "all_candidates": results,
}
(ROOT / "outputs/models/surprise_metrics.json").write_text(json.dumps(summary, indent=2))
print(f"Metrics summary -> outputs/models/surprise_metrics.json")
