"""Save XGBoost val/test predictions to CSV and compute test NRMSE."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

r = json.load(open(ROOT / "outputs/models/xgb_result.json"))

# Save val preds
val_df = pd.DataFrame({
    "timestamp": pd.to_datetime(r["val_timestamps"]),
    "load_pred": r["val_preds"],
})
val_df.to_csv(ROOT / "outputs/forecasts/xgb_val_preds.csv", index=False)

# Save test preds
test_df = pd.DataFrame({
    "timestamp": pd.to_datetime(r["test_timestamps"]),
    "load_pred": r["test_preds"],
})
test_df.to_csv(ROOT / "outputs/forecasts/xgb_test_preds.csv", index=False)

# Compute test NRMSE
features_test = pd.read_parquet(ROOT / "data/features/features_test.parquet")
features_test["timestamp"] = pd.to_datetime(features_test["timestamp"])
merged = features_test[["timestamp", "load_kw"]].merge(test_df, on="timestamp", how="left")
y_true = merged["load_kw"].values
y_pred = merged["load_pred"].values
mask = ~np.isnan(y_pred)
y_true = y_true[mask]
y_pred = y_pred[mask]

rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
mae  = float(np.mean(np.abs(y_true - y_pred)))
nrmse = rmse / y_true.mean() * 100

print(f"XGBoost on Apr+Sep 2025 test:")
print(f"  N         : {len(y_true)}")
print(f"  RMSE      : {rmse:.4f} kW")
print(f"  MAE       : {mae:.4f} kW")
print(f"  NRMSE     : {nrmse:.2f}%")
print(f"  mean(load): {y_true.mean():.4f} kW")

# Per-month
for mo, name in [(4, "April"), (9, "September")]:
    m = merged[(merged["timestamp"].dt.month == mo) & (~merged["load_pred"].isna())]
    yt, yp = m["load_kw"].values, m["load_pred"].values
    rmse_m = float(np.sqrt(np.mean((yt - yp)**2)))
    nrmse_m = rmse_m / yt.mean() * 100
    print(f"  {name}: NRMSE={nrmse_m:.2f}%  RMSE={rmse_m:.4f}")
