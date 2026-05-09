"""
The diagnostics show severe under-prediction of high-load spikes.
Try log-transform target (and asymmetric loss) to give the model more
incentive to predict spikes correctly.

Approaches:
  A. target = log(load + EPS), regress, then exp - EPS
  B. target = sqrt(load)
  C. weight high-load samples in training
  D. quantile regression at quantile=0.6 (slight upward bias)
  E. ensemble: 0.5 * (mean model) + 0.5 * (high-quantile model)
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

EPS = 0.1
def nrmse(y, yp): return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)
def rmse(y, yp):  return float(np.sqrt(np.mean((y - yp) ** 2)))
def mae(y, yp):   return float(np.mean(np.abs(y - yp)))

BASE = dict(
    n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
    min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    verbose=-1, n_jobs=-1, random_state=42,
)

# Reference: huber model on raw target
print("Reference: LGBM huber on raw load")
m0 = lgb.LGBMRegressor(**BASE, objective="huber", alpha=0.9)
m0.fit(X_tv, y_tv)
test_p0 = np.clip(m0.predict(X_te), 0, None)
print(f"  test RMSE={rmse(y_te,test_p0):.4f}  NRMSE={nrmse(y_te,test_p0):.2f}%  MAE={mae(y_te,test_p0):.4f}")

# A. log(load + EPS) target
print("\nA. log target")
m_log = lgb.LGBMRegressor(**BASE, objective="regression")
m_log.fit(X_tv, np.log(y_tv + EPS))
log_p = m_log.predict(X_te)
test_p_log = np.clip(np.exp(log_p) - EPS, 0, None)
print(f"  test RMSE={rmse(y_te,test_p_log):.4f}  NRMSE={nrmse(y_te,test_p_log):.2f}%  MAE={mae(y_te,test_p_log):.4f}")

# B. sqrt(load) target
print("\nB. sqrt target")
m_sq = lgb.LGBMRegressor(**BASE, objective="regression")
m_sq.fit(X_tv, np.sqrt(y_tv))
sq_p = m_sq.predict(X_te)
test_p_sq = np.clip(sq_p ** 2, 0, None)
print(f"  test RMSE={rmse(y_te,test_p_sq):.4f}  NRMSE={nrmse(y_te,test_p_sq):.2f}%  MAE={mae(y_te,test_p_sq):.4f}")

# C. weight high-load samples (inverse density)
print("\nC. weighted by load-quantile (upweight rare high loads)")
# Bin train+val by load and inverse-frequency weight
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
y_bin = discretizer.fit_transform(y_tv.reshape(-1, 1)).flatten().astype(int)
counts = np.bincount(y_bin)
inv_freq = 1.0 / counts
w = inv_freq[y_bin]
w = w / w.mean()  # normalize to mean=1
m_w = lgb.LGBMRegressor(**BASE, objective="huber", alpha=0.9)
m_w.fit(X_tv, y_tv, sample_weight=w)
test_p_w = np.clip(m_w.predict(X_te), 0, None)
print(f"  test RMSE={rmse(y_te,test_p_w):.4f}  NRMSE={nrmse(y_te,test_p_w):.2f}%  MAE={mae(y_te,test_p_w):.4f}")

# D. quantile=0.55 (very slight upward push to capture spikes)
print("\nD. quantile regression alpha=0.55")
m_q = lgb.LGBMRegressor(**BASE, objective="quantile", alpha=0.55)
m_q.fit(X_tv, y_tv)
test_p_q = np.clip(m_q.predict(X_te), 0, None)
print(f"  test RMSE={rmse(y_te,test_p_q):.4f}  NRMSE={nrmse(y_te,test_p_q):.2f}%  MAE={mae(y_te,test_p_q):.4f}")

# E. ensemble: huber + log + quantile_0.55
print("\nE. ensemble huber + log + q0.55 + sqrt (mean)")
test_p_ens = (test_p0 + test_p_log + test_p_q + test_p_sq) / 4
print(f"  test RMSE={rmse(y_te,test_p_ens):.4f}  NRMSE={nrmse(y_te,test_p_ens):.2f}%  MAE={mae(y_te,test_p_ens):.4f}")

# F. weighted mean: optimize on val
print("\nF. blend weights optimized on val")
val_p0 = np.clip(m0.predict(X_va), 0, None)
val_p_log = np.clip(np.exp(m_log.predict(X_va)) - EPS, 0, None)
val_p_sq  = np.clip(m_sq.predict(X_va) ** 2, 0, None)
val_p_q   = np.clip(m_q.predict(X_va), 0, None)
val_p_w   = np.clip(m_w.predict(X_va), 0, None)
P_va = np.column_stack([val_p0, val_p_log, val_p_sq, val_p_q, val_p_w])
P_te = np.column_stack([test_p0, test_p_log, test_p_sq, test_p_q, test_p_w])
from scipy.optimize import minimize
def loss_fn(w):
    w = np.maximum(w, 0); w /= max(w.sum(), 1e-6)
    return np.mean((y_va - P_va @ w) ** 2)
res = minimize(loss_fn, np.ones(5)/5, method="Nelder-Mead")
w_opt = np.maximum(res.x, 0); w_opt /= w_opt.sum()
test_p_blnd = P_te @ w_opt
print(f"  weights: {dict(zip(['huber','log','sqrt','q55','wgt'], np.round(w_opt, 3)))}")
print(f"  test RMSE={rmse(y_te,test_p_blnd):.4f}  NRMSE={nrmse(y_te,test_p_blnd):.2f}%  MAE={mae(y_te,test_p_blnd):.4f}")

# Check error in HIGH-LOAD bins for top approaches
print("\nERROR IN HIGH-LOAD BINS (the failure mode)")
high_mask = y_te >= 1.5
print(f"  N in high-load bin (>=1.5kW): {high_mask.sum()}")
for label, p in [("huber raw", test_p0), ("log", test_p_log), ("sqrt", test_p_sq),
                  ("q55", test_p_q), ("weighted", test_p_w), ("blend", test_p_blnd)]:
    print(f"  {label:12s}  high-RMSE = {rmse(y_te[high_mask], p[high_mask]):.4f}  "
          f"high-MAE = {mae(y_te[high_mask], p[high_mask]):.4f}")

# Save best
out = ROOT / "outputs/forecasts/lgbm_logblend_test_preds.csv"
pd.DataFrame({"timestamp": test_df["timestamp"], "load_pred": test_p_blnd}).to_csv(out, index=False)
print(f"\nSaved blend -> {out}")
