"""
Retrain LightGBM locally with best hyperparams from Modal run.
Saves model for live inference in MPC.
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle

ROOT  = Path(__file__).parents[1]
FEAT  = ROOT / "data" / "features"
MDIR  = ROOT / "outputs" / "models"

# Load best hyperparams from Modal run
r = json.loads((MDIR / "lgbm_result.json").read_text())
best_params = r["best_params"]
best_params.update({"verbose": -1, "n_jobs": -1, "random_state": 42})

# Feature columns (matching Modal script)
FEATURE_COLS = [
    "lag_1","lag_4","lag_8","lag_96","lag_192","lag_672",
    "pv_lag1","pv_lag96",
    "roll_4_mean","roll_16_mean","roll_96_mean","roll_4_std","roll_96_std",
    "net_load_lag1","net_load_lag96",
    "hour","dow","month","day_of_year","is_weekend","is_holiday","tariff_enc",
    "buy_price",
    "sin_24h","cos_24h","sin_12h","cos_12h","sin_8h","cos_8h",
    "sin_annual","cos_annual",
    "temperature_2m","shortwave_radiation","cloud_cover","relative_humidity_2m",
    "hdd","cdd","temp_lag96","rad_lag96",
]
TARGET = "load_kw"

print("Loading features...")
train_df = pd.read_parquet(FEAT / "features_train.parquet")
val_df   = pd.read_parquet(FEAT / "features_val.parquet")
test_df  = pd.read_parquet(FEAT / "features_test.parquet")

avail = [c for c in FEATURE_COLS if c in train_df.columns]
print(f"  Features: {len(avail)}")

# Train on train+val combined for best generalization (val months match test months)
trainval_df = pd.concat([train_df, val_df], ignore_index=True)
X_tv, y_tv = trainval_df[avail].values, trainval_df[TARGET].values
X_tr, y_tr = train_df[avail].values, train_df[TARGET].values
X_va, y_va = val_df[avail].values,   val_df[TARGET].values
X_te       = test_df[avail].values

print(f"  Train: {len(X_tr)}  Val: {len(X_va)}  TrainVal: {len(X_tv)}  Test: {len(X_te)}")

t0 = time.time()
print("Training LightGBM on train+val with fixed n_estimators (no early stopping)...")
model = lgb.LGBMRegressor(**best_params)
model.fit(
    X_tv, y_tv,
    callbacks=[lgb.log_evaluation(200)],
)
elapsed = time.time() - t0

def nrmse(y_t, y_p):
    return float(np.sqrt(np.mean((y_t - y_p)**2)) / np.mean(y_t) * 100)

val_preds  = model.predict(X_va)
test_preds = model.predict(X_te)

y_te = test_df[TARGET].values

val_n  = nrmse(y_va, val_preds)
test_n = nrmse(y_te, test_preds)
print(f"\nTraining done in {elapsed:.1f}s")
print(f"Val  NRMSE (Apr+Sep 2024): {val_n:.3f}%")
print(f"Test NRMSE (Apr+Sep 2025): {test_n:.3f}%")
print(f"Val  RMSE  : {np.sqrt(np.mean((y_va-val_preds)**2)):.4f} kW")
print(f"Val  MAE   : {np.mean(np.abs(y_va-val_preds)):.4f} kW")

# Save model
model_path = MDIR / "lgbm_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "feature_cols": avail}, f)
print(f"\nModel saved -> {model_path}")

# Save val and test preds
pred_dir = ROOT / "outputs" / "forecasts"
pred_dir.mkdir(exist_ok=True)

pd.DataFrame({"timestamp": val_df["timestamp"], "load_pred": val_preds}).to_csv(pred_dir / "lgbm_val_preds.csv", index=False)
pd.DataFrame({"timestamp": test_df["timestamp"], "load_pred": test_preds}).to_csv(pred_dir / "lgbm_test_preds.csv", index=False)
print(f"Predictions saved to {pred_dir}")
