"""Quick local test of v2 features with previous best LGBM hyperparams."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import time

ROOT = Path(__file__).parents[1]

# Load v2
train = pd.read_parquet(ROOT / "data/features/features_v2_train.parquet")
val   = pd.read_parquet(ROOT / "data/features/features_v2_val.parquet")
test  = pd.read_parquet(ROOT / "data/features/features_v2_test.parquet")

DROP = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
        "qow","hod","net_load","sell_price","pv_today_total"}
feats = [c for c in train.columns if c not in DROP]
print(f"v2 features: {len(feats)}")
print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

X_tr, y_tr = train[feats].values, train["load_kw"].values
X_va, y_va = val[feats].values,   val["load_kw"].values
X_te, y_te = test[feats].values,  test["load_kw"].values
# Combined train+val for final fit
X_tv = np.vstack([X_tr, X_va])
y_tv = np.concatenate([y_tr, y_va])


def nrmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)


# ── LightGBM with previous best params (as a sanity baseline) ────
r1 = json.load(open(ROOT / "outputs/models/lgbm_result.json"))
params = r1["best_params"].copy()
params.update({"verbose": -1, "n_jobs": -1, "random_state": 42})

print("\n=== LightGBM v2 (prev best hyperparams) ===")
t0 = time.time()
m = lgb.LGBMRegressor(**params)
m.fit(X_tv, y_tv)
val_p  = m.predict(X_va)
test_p = m.predict(X_te)
print(f"  Wall: {time.time()-t0:.1f}s")
print(f"  Val  NRMSE: {nrmse(y_va, val_p):.2f}%")
print(f"  Test NRMSE: {nrmse(y_te, test_p):.2f}%")
# Top features
imp = m.feature_importances_
top = sorted(enumerate(imp), key=lambda x: -x[1])[:15]
print("  Top 15 features:")
for i, v in top:
    print(f"    {feats[i]:25s}  {v}")

# ── Quick XGBoost ────────────────────────────────────────────────
print("\n=== XGBoost v2 ===")
t0 = time.time()
mx = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=8, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, tree_method="hist",
)
mx.fit(X_tv, y_tv)
val_p2  = mx.predict(X_va)
test_p2 = mx.predict(X_te)
print(f"  Wall: {time.time()-t0:.1f}s")
print(f"  Val  NRMSE: {nrmse(y_va, val_p2):.2f}%")
print(f"  Test NRMSE: {nrmse(y_te, test_p2):.2f}%")

# ── Simple ensemble of LGBM + XGB ────────────────────────────────
ens_test = 0.5 * test_p + 0.5 * test_p2
print(f"\n=== Ensemble (LGBM+XGB v2) ===")
print(f"  Test NRMSE: {nrmse(y_te, ens_test):.2f}%")

# Save predictions in case they're useful
out = ROOT / "outputs/forecasts"
pd.DataFrame({"timestamp": test["timestamp"], "load_pred": test_p}).to_csv(out / "lgbm_v2_test_preds.csv", index=False)
pd.DataFrame({"timestamp": test["timestamp"], "load_pred": test_p2}).to_csv(out / "xgb_v2_test_preds.csv", index=False)
pd.DataFrame({"timestamp": test["timestamp"], "load_pred": ens_test}).to_csv(out / "ensemble_v2_test_preds.csv", index=False)
print(f"\nSaved -> {out}")
