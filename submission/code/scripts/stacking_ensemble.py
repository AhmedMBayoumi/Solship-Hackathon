"""
Stacking ensemble: LGBM + XGB + CatBoost + Ridge meta-learner.
Plus deep MLP. Target: reduce variance beyond what any single model gets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import time

ROOT = Path(__file__).parents[1]

train_df = pd.read_parquet(ROOT / "data/features/features_v2_train.parquet")
val_df   = pd.read_parquet(ROOT / "data/features/features_v2_val.parquet")
test_df  = pd.read_parquet(ROOT / "data/features/features_v2_test.parquet")

DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
             "qow","hod","net_load","sell_price","pv_today_total"}
DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
feats = [c for c in train_df.columns if c not in (DROP_BASE | DROP_LEAKY)]

X_tr, y_tr = train_df[feats].values, train_df["load_kw"].values
X_va, y_va = val_df[feats].values,   val_df["load_kw"].values
X_te, y_te = test_df[feats].values,  test_df["load_kw"].values
X_tv = np.vstack([X_tr, X_va])
y_tv = np.concatenate([y_tr, y_va])

print(f"Features: {len(feats)}  Train+Val: {len(X_tv)}  Test: {len(X_te)}")

def nrmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)

def metrics(y, yp, label):
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    mae  = float(np.mean(np.abs(y - yp)))
    n    = nrmse(y, yp)
    r2   = 1 - np.sum((y-yp)**2) / np.sum((y - y.mean())**2)
    print(f"  {label:35s}  RMSE={rmse:.4f}  MAE={mae:.4f}  NRMSE={n:.2f}%  R2={r2:.3f}")
    return n

# ── Train base models on train only, predict val (for stacking labels) and test
print("\n=== Base models (trained on TRAIN, predict VAL & TEST) ===")
print("These are the level-0 predictions for stacking.\n")
HUBER = {"objective":"huber", "alpha":0.9}

# 1. LGBM — heavy reg
t0 = time.time()
lgbm = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
    min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    objective="huber", alpha=0.9,
    verbose=-1, n_jobs=-1, random_state=42,
)
lgbm.fit(X_tr, y_tr)
val_lgbm  = np.clip(lgbm.predict(X_va), 0, None)
test_lgbm = np.clip(lgbm.predict(X_te), 0, None)
metrics(y_va, val_lgbm,  "LGBM val")
metrics(y_te, test_lgbm, "LGBM test")
print(f"  ({time.time()-t0:.0f}s)")

# 2. XGBoost — heavy reg
t0 = time.time()
xgbm = xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.01, max_depth=4,
    min_child_weight=20, reg_alpha=2.0, reg_lambda=3.0,
    subsample=0.7, colsample_bytree=0.7,
    n_jobs=-1, random_state=42, tree_method="hist",
    objective="reg:pseudohubererror", huber_slope=0.5,
)
xgbm.fit(X_tr, y_tr)
val_xgb  = np.clip(xgbm.predict(X_va), 0, None)
test_xgb = np.clip(xgbm.predict(X_te), 0, None)
metrics(y_va, val_xgb,  "XGB val")
metrics(y_te, test_xgb, "XGB test")
print(f"  ({time.time()-t0:.0f}s)")

# 3. CatBoost
t0 = time.time()
from catboost import CatBoostRegressor
CAT_FEATS = ["dow","month","hour","is_weekend","is_holiday","tariff_enc","is_high_pv_day"]
cat_idx = [feats.index(c) for c in CAT_FEATS if c in feats]
cat = CatBoostRegressor(
    iterations=3000, learning_rate=0.02, depth=6,
    l2_leaf_reg=10.0, random_seed=42, verbose=0,
    bagging_temperature=2.0, subsample=0.7,
    cat_features=cat_idx,
    loss_function="Huber:delta=0.5",
)
def to_cat_df(X):
    df_x = pd.DataFrame(X, columns=feats)
    for c in CAT_FEATS:
        if c in df_x.columns:
            df_x[c] = df_x[c].astype(int)
    return df_x

cat.fit(to_cat_df(X_tr), y_tr, verbose=0)
val_cat  = np.clip(cat.predict(to_cat_df(X_va)), 0, None)
test_cat = np.clip(cat.predict(to_cat_df(X_te)), 0, None)
metrics(y_va, val_cat,  "CAT val")
metrics(y_te, test_cat, "CAT test")
print(f"  ({time.time()-t0:.0f}s)")

# Simple averaging
print("\n=== Simple averaging ===")
val_avg  = (val_lgbm + val_xgb + val_cat) / 3
test_avg = (test_lgbm + test_xgb + test_cat) / 3
metrics(y_va, val_avg,  "Average val")
metrics(y_te, test_avg, "Average test")

# Stacking with Ridge meta-learner
print("\n=== Stacked (Ridge meta-learner on val preds) ===")
from sklearn.linear_model import Ridge
P_va = np.column_stack([val_lgbm, val_xgb, val_cat])
P_te = np.column_stack([test_lgbm, test_xgb, test_cat])
meta = Ridge(alpha=1.0).fit(P_va, y_va)
print(f"  meta weights: {meta.coef_}  intercept: {meta.intercept_:.4f}")
val_stack  = np.clip(meta.predict(P_va), 0, None)
test_stack = np.clip(meta.predict(P_te), 0, None)
metrics(y_va, val_stack,  "Stacked val")
metrics(y_te, test_stack, "Stacked test")

# Constrained non-negative meta (better for generalization)
from scipy.optimize import minimize
def loss(w):
    p = P_va @ np.maximum(w, 0)
    p /= max(np.sum(np.maximum(w, 0)), 1e-6)
    return np.mean((y_va - p) ** 2)
res = minimize(loss, np.array([1, 1, 1]), method="Nelder-Mead")
w = np.maximum(res.x, 0); w /= w.sum()
val_blend  = np.clip(P_va @ w, 0, None)
test_blend = np.clip(P_te @ w, 0, None)
print(f"\n=== Convex blend  weights: {w} ===")
metrics(y_va, val_blend,  "Blend val")
metrics(y_te, test_blend, "Blend test")

# ── Final: refit each model on train+val, pick best ensemble
print("\n=== REFIT on train+val, save best ===")
lgbm.fit(X_tv, y_tv)
xgbm.fit(X_tv, y_tv)
cat.fit(to_cat_df(X_tv), y_tv, verbose=0)
test_final_lgbm = np.clip(lgbm.predict(X_te), 0, None)
test_final_xgb  = np.clip(xgbm.predict(X_te), 0, None)
test_final_cat  = np.clip(cat.predict(to_cat_df(X_te)), 0, None)
test_final_avg  = (test_final_lgbm + test_final_xgb + test_final_cat) / 3
test_final_blnd = np.column_stack([test_final_lgbm, test_final_xgb, test_final_cat]) @ w

metrics(y_te, test_final_lgbm, "FINAL LGBM (retrained)")
metrics(y_te, test_final_xgb,  "FINAL XGB  (retrained)")
metrics(y_te, test_final_cat,  "FINAL CAT  (retrained)")
metrics(y_te, test_final_avg,  "FINAL avg  (retrained)")
metrics(y_te, test_final_blnd, "FINAL blend (retrained)")

# Save predictions
out = ROOT / "outputs/forecasts"
pd.DataFrame({"timestamp": test_df["timestamp"], "load_pred": test_final_blnd}).to_csv(
    out / "stacked_v2_test_preds.csv", index=False)
print(f"\nSaved -> {out}/stacked_v2_test_preds.csv")
