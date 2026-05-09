"""Local CatBoost training with v2 features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import time

ROOT = Path(__file__).parents[1]

train = pd.read_parquet(ROOT / "data/features/features_v2_train.parquet")
val   = pd.read_parquet(ROOT / "data/features/features_v2_val.parquet")
test  = pd.read_parquet(ROOT / "data/features/features_v2_test.parquet")

DROP = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
        "qow","hod","net_load","sell_price","pv_today_total",
        "qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
feats = [c for c in train.columns if c not in DROP]
print(f"Features: {len(feats)}")

CAT_FEATS = ["dow","month","hour","is_weekend","is_holiday","tariff_enc","is_high_pv_day"]
cat_idx = [feats.index(c) for c in CAT_FEATS if c in feats]

X_tr, y_tr = train[feats], train["load_kw"].values
X_va, y_va = val[feats],   val["load_kw"].values
X_te, y_te = test[feats],  test["load_kw"].values
X_tv = pd.concat([X_tr, X_va], ignore_index=True)
y_tv = np.concatenate([y_tr, y_va])

def nrmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)

try:
    from catboost import CatBoostRegressor
except ImportError:
    print("Installing catboost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostRegressor

print("\n=== CatBoost — moderate ===")
t0 = time.time()
m = CatBoostRegressor(
    iterations=3000, learning_rate=0.03, depth=6,
    l2_leaf_reg=5.0, random_seed=42, verbose=0,
    bagging_temperature=1.0, od_type="Iter", od_wait=100,
    cat_features=cat_idx,
)
m.fit(X_tv.copy(), y_tv, eval_set=(X_va.copy(), y_va), use_best_model=True, verbose=0)
val_p  = np.clip(m.predict(X_va), 0, None)
test_p = np.clip(m.predict(X_te), 0, None)
print(f"  Wall: {time.time()-t0:.1f}s  best_iter: {m.best_iteration_}")
print(f"  Val  NRMSE: {nrmse(y_va, val_p):.2f}%")
print(f"  Test NRMSE: {nrmse(y_te, test_p):.2f}%")

print("\n=== CatBoost — heavy reg (depth=4, l2=10) ===")
t0 = time.time()
m2 = CatBoostRegressor(
    iterations=5000, learning_rate=0.01, depth=4,
    l2_leaf_reg=10.0, random_seed=42, verbose=0,
    bagging_temperature=2.0, subsample=0.7,
    cat_features=cat_idx,
)
m2.fit(X_tv.copy(), y_tv, verbose=0)
val_p2  = np.clip(m2.predict(X_va), 0, None)
test_p2 = np.clip(m2.predict(X_te), 0, None)
print(f"  Wall: {time.time()-t0:.1f}s")
print(f"  Val  NRMSE: {nrmse(y_va, val_p2):.2f}%")
print(f"  Test NRMSE: {nrmse(y_te, test_p2):.2f}%")

# Save
out = ROOT / "outputs/forecasts/catboost_v2_test_preds.csv"
pd.DataFrame({"timestamp": test["timestamp"], "load_pred": test_p}).to_csv(out, index=False)
print(f"\nSaved CatBoost moderate -> {out}")
