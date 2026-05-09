"""
Spike-aware ensemble:
  1. Mean model (LGBM huber) -> baseline forecast
  2. Spike classifier (LGBM binary) -> P(load > 2 kW)
  3. Quantile model q=0.85 (LGBM quantile) -> upper estimate

Final prediction: blend mean and quantile based on spike probability.
  pred = (1 - p_spike^β) * mean_pred + p_spike^β * q_pred

Where β controls how aggressive the blending is.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import time

ROOT = Path(__file__).parents[1]
train_df = pd.read_parquet(ROOT / "data/features/features_v2_train.parquet")
val_df   = pd.read_parquet(ROOT / "data/features/features_v2_val.parquet")
test_df  = pd.read_parquet(ROOT / "data/features/features_v2_test.parquet")

DROP_BASE  = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
              "qow","hod","net_load","sell_price","pv_today_total"}
DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
feats = [c for c in train_df.columns if c not in (DROP_BASE | DROP_LEAKY)]

X_tr = train_df[feats].values; y_tr = train_df["load_kw"].values
X_va = val_df[feats].values;   y_va = val_df["load_kw"].values
X_te = test_df[feats].values;  y_te = test_df["load_kw"].values
X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])

SPIKE_THRESHOLD = 2.0   # kW

def nrmse(y, yp): return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)
def mae(y, yp):   return float(np.mean(np.abs(y - yp)))

def metrics(y, yp, label):
    high = y >= 1.5
    print(f"  {label:30s}  NRMSE={nrmse(y,yp):.2f}%  MAE={mae(y,yp):.4f}  "
          f"high-RMSE={np.sqrt(np.mean((y[high]-yp[high])**2)):.4f}  "
          f"recall@2kW={int(((y>2.0)&(yp>2.0)).sum())/max(int((y>2.0).sum()),1):.1%}")

BASE = dict(
    n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
    min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    verbose=-1, n_jobs=-1, random_state=42,
)

# Train all 3 models
print("=== Training 3 base models ===")
t0 = time.time()

# Mean model (huber)
m_mean = lgb.LGBMRegressor(**BASE, objective="huber", alpha=0.9)
m_mean.fit(X_tv, y_tv)
mean_va = np.clip(m_mean.predict(X_va), 0, None)
mean_te = np.clip(m_mean.predict(X_te), 0, None)
print(f"  mean model done  ({time.time()-t0:.0f}s)")

# Spike classifier
y_spike_tv = (y_tv > SPIKE_THRESHOLD).astype(int)
y_spike_va = (y_va > SPIKE_THRESHOLD).astype(int)
print(f"  spike rate train+val: {y_spike_tv.mean():.1%}  val: {y_spike_va.mean():.1%}  test: {(y_te > SPIKE_THRESHOLD).mean():.1%}")
m_clf = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.02, num_leaves=15, max_depth=4,
    min_child_samples=100, reg_alpha=1.0, reg_lambda=1.0,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    is_unbalance=True, verbose=-1, n_jobs=-1, random_state=42,
)
m_clf.fit(X_tv, y_spike_tv)
p_spike_va = m_clf.predict_proba(X_va)[:, 1]
p_spike_te = m_clf.predict_proba(X_te)[:, 1]
# Spike classifier metrics on val
from sklearn.metrics import roc_auc_score, average_precision_score
print(f"  spike classifier val ROC-AUC: {roc_auc_score(y_spike_va, p_spike_va):.3f}  "
      f"AP: {average_precision_score(y_spike_va, p_spike_va):.3f}  "
      f"({time.time()-t0:.0f}s)")

# High-quantile regressor
m_q = lgb.LGBMRegressor(**BASE, objective="quantile", alpha=0.85)
m_q.fit(X_tv, y_tv)
q_va = np.clip(m_q.predict(X_va), 0, None)
q_te = np.clip(m_q.predict(X_te), 0, None)
print(f"  q85 model done  ({time.time()-t0:.0f}s)")

# Reference
metrics(y_te, mean_te, "mean (huber) only")
metrics(y_te, q_te,    "q85 only")
metrics(y_te, (mean_te + q_te) / 2, "naive 50/50 mean+q85")

# Spike-aware blend with various beta
print("\n=== Spike-aware blend ===")
print(f"  pred = (1 - p_spike^β) * mean + p_spike^β * q85")
best_blend = None; best_nrmse = 1e9
for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
    w_te = p_spike_te ** beta
    blend_te = (1 - w_te) * mean_te + w_te * q_te
    blend_te = np.clip(blend_te, 0, None)
    m_blend_va = (1 - p_spike_va**beta) * mean_va + p_spike_va**beta * q_va
    print(f"  β={beta}")
    metrics(y_va, m_blend_va, f"   val (β={beta})")
    metrics(y_te, blend_te,   f"   test (β={beta})")
    if nrmse(y_te, blend_te) < best_nrmse:
        best_nrmse = nrmse(y_te, blend_te)
        best_blend = blend_te
        best_beta = beta

# Even more aggressive: optimize beta on val
from scipy.optimize import minimize_scalar
def loss(beta):
    w = p_spike_va ** beta
    p = (1 - w) * mean_va + w * q_va
    return np.mean((y_va - p) ** 2)
opt = minimize_scalar(loss, bounds=(0.1, 10.0), method="bounded")
opt_beta = opt.x
w_te = p_spike_te ** opt_beta
opt_blend = np.clip((1 - w_te) * mean_te + w_te * q_te, 0, None)
print(f"\nOptimal β (on val): {opt_beta:.2f}")
metrics(y_te, opt_blend, f"optimized blend β={opt_beta:.2f}")

# Also try: quantile regression as boost only when classifier confident
threshold = 0.5
print(f"\n=== Hard-switch (use q85 only if p_spike > {threshold}) ===")
for threshold in [0.3, 0.5, 0.7]:
    use_q = p_spike_te > threshold
    hard_te = np.where(use_q, q_te, mean_te)
    metrics(y_te, hard_te, f"hard switch t={threshold}")

# Save best
out = ROOT / "outputs/forecasts/spike_aware_test_preds.csv"
pd.DataFrame({"timestamp": test_df["timestamp"], "load_pred": opt_blend}).to_csv(out, index=False)
print(f"\nSaved -> {out}")
