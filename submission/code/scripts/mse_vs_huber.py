"""
Test the user's question: are huber + heavy regularization holding us back?
Compare:
  A. huber + heavy reg (current)
  B. mse   + heavy reg
  C. huber + light reg
  D. mse   + light reg
  E. mse   + NO reg (let it overfit val)

Reports: val, test, val-test gap, MAE, high-load RMSE.
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

def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)
def mae(y, yp):   return float(np.mean(np.abs(y - yp)))
def high_rmse(y, yp, thr=1.5):
    m = y >= thr
    return float(np.sqrt(np.mean((y[m]-yp[m])**2)))

CONFIGS = [
    {"label":"A. huber + heavy reg (current)",
     "params": dict(n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
                    min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
                    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                    objective="huber", alpha=0.9)},
    {"label":"B. mse   + heavy reg",
     "params": dict(n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
                    min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
                    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                    objective="regression")},
    {"label":"C. huber + light reg",
     "params": dict(n_estimators=2000, learning_rate=0.02, num_leaves=63, max_depth=8,
                    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                    subsample=0.9, colsample_bytree=0.9,
                    objective="huber", alpha=0.9)},
    {"label":"D. mse   + light reg",
     "params": dict(n_estimators=2000, learning_rate=0.02, num_leaves=63, max_depth=8,
                    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                    subsample=0.9, colsample_bytree=0.9,
                    objective="regression")},
    {"label":"E. mse   + NO reg (deep)",
     "params": dict(n_estimators=2000, learning_rate=0.03, num_leaves=255, max_depth=-1,
                    min_child_samples=5,
                    objective="regression")},
    {"label":"F. mse   + medium reg",
     "params": dict(n_estimators=2500, learning_rate=0.015, num_leaves=31, max_depth=6,
                    min_child_samples=50, reg_alpha=0.5, reg_lambda=1.0,
                    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                    objective="regression")},
]

print(f"{'config':<37s}  {'val':>6s}  {'test':>6s}  {'gap':>6s}  {'MAE':>7s}  {'highRMSE':>8s}")
print("-" * 80)
preds = {}
for c in CONFIGS:
    p = dict(c["params"])
    p.update({"verbose": -1, "n_jobs": -1, "random_state": 42})
    t0 = time.time()
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tv, y_tv)
    val_p  = np.clip(m.predict(X_va), 0, None)
    test_p = np.clip(m.predict(X_te), 0, None)
    nv = nrmse(y_va, val_p); nt = nrmse(y_te, test_p)
    print(f"  {c['label']:<35s}  {nv:>5.2f}%  {nt:>5.2f}%  {nt-nv:>+5.2f}  {mae(y_te,test_p):>7.4f}  {high_rmse(y_te,test_p):>8.4f}  ({time.time()-t0:.0f}s)")
    preds[c["label"]] = test_p

# Pick best-test
best_label = min(preds, key=lambda k: nrmse(y_te, preds[k]))
print(f"\nBest test NRMSE: {best_label}  ({nrmse(y_te, preds[best_label]):.2f}%)")
out = ROOT / "outputs/forecasts/lgbm_best_loss_test_preds.csv"
pd.DataFrame({"timestamp": test_df["timestamp"], "load_pred": preds[best_label]}).to_csv(out, index=False)
print(f"Saved -> {out}")
