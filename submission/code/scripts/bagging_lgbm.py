"""
Bagging: train N LightGBM models with different random seeds + bootstrap
samples, average predictions. Reduces variance via diversity.
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

def nrmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)

N_BAGS = 12
print(f"Bagging {N_BAGS} LGBMs with diverse hyperparams + seeds...\n")

val_preds = np.zeros((N_BAGS, len(y_va)))
test_preds = np.zeros((N_BAGS, len(y_te)))
configs = [
    {"num_leaves": 15, "max_depth": 4,  "learning_rate": 0.01, "min_child_samples": 100, "reg_alpha": 2.0, "reg_lambda": 3.0, "subsample": 0.7, "colsample_bytree": 0.7},
    {"num_leaves": 31, "max_depth": 5,  "learning_rate": 0.02, "min_child_samples": 50,  "reg_alpha": 1.0, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"num_leaves": 7,  "max_depth": 3,  "learning_rate": 0.005,"min_child_samples": 200, "reg_alpha": 5.0, "reg_lambda": 5.0, "subsample": 0.6, "colsample_bytree": 0.6},
    {"num_leaves": 63, "max_depth": 6,  "learning_rate": 0.03, "min_child_samples": 30,  "reg_alpha": 0.5, "reg_lambda": 0.5, "subsample": 0.9, "colsample_bytree": 0.9},
]

t0 = time.time()
for i in range(N_BAGS):
    cfg = configs[i % len(configs)]
    seed = 42 + i
    p = dict(cfg)
    p.update({
        "n_estimators": 3000, "subsample_freq": 1,
        "objective": "huber", "alpha": 0.9,
        "verbose": -1, "n_jobs": -1, "random_state": seed,
    })
    rng = np.random.default_rng(seed)
    # Bootstrap row sample 90% (with replacement)
    idx = rng.integers(0, len(X_tr), size=int(len(X_tr) * 0.9))
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr[idx], y_tr[idx])
    val_preds[i]  = np.clip(m.predict(X_va), 0, None)
    test_preds[i] = np.clip(m.predict(X_te), 0, None)
    print(f"  [{i+1}/{N_BAGS}] cfg={cfg['num_leaves']}/{cfg['max_depth']}  seed={seed}  "
          f"val={nrmse(y_va, val_preds[i]):.2f}%  test={nrmse(y_te, test_preds[i]):.2f}%")

print(f"\nTotal: {time.time()-t0:.0f}s")

# Averages
val_avg  = val_preds.mean(axis=0)
test_avg = test_preds.mean(axis=0)
val_med  = np.median(val_preds, axis=0)
test_med = np.median(test_preds, axis=0)

print(f"\n{'method':>20s}  {'val NRMSE':>10s}  {'test NRMSE':>11s}")
print(f"  {'best individual':>20s}  {min(nrmse(y_va, p) for p in val_preds):>9.2f}%  {min(nrmse(y_te, p) for p in test_preds):>10.2f}%")
print(f"  {'bag mean':>20s}  {nrmse(y_va, val_avg):>9.2f}%  {nrmse(y_te, test_avg):>10.2f}%")
print(f"  {'bag median':>20s}  {nrmse(y_va, val_med):>9.2f}%  {nrmse(y_te, test_med):>10.2f}%")

out = ROOT / "outputs/forecasts/bagging_test_preds.csv"
pd.DataFrame({"timestamp": test_df["timestamp"], "load_pred": test_avg}).to_csv(out, index=False)
print(f"Saved -> {out}")
