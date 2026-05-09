import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Ahmed Bayoumi\University\ZC Hackathon")
r = json.loads((ROOT / "outputs/models/lgbm_result.json").read_text())

pred_dir = ROOT / "outputs" / "forecasts"
pred_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"timestamp": r["val_timestamps"], "load_pred": r["val_preds"]}).to_csv(pred_dir / "lgbm_val_preds.csv", index=False)
pd.DataFrame({"timestamp": r["test_timestamps"], "load_pred": r["test_preds"]}).to_csv(pred_dir / "lgbm_test_preds.csv", index=False)

df_2025 = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_test = df_2025[(df_2025["timestamp"].dt.year == 2025) & df_2025["timestamp"].dt.month.isin([4, 9])]

preds = pd.read_csv(pred_dir / "lgbm_test_preds.csv", parse_dates=["timestamp"])
merged = df_test.merge(preds, on="timestamp", how="inner")

rmse_test = float(np.sqrt(np.mean((merged["load_kw"] - merged["load_pred"])**2)))
nrmse_test = rmse_test / merged["load_kw"].mean() * 100
mae_test = float(np.mean(np.abs(merged["load_kw"] - merged["load_pred"])))

print(f"LightGBM RESULTS:")
print(f"  Val  NRMSE (Apr+Sep 2024): {r['final_val_nrmse']:.3f}%")
print(f"  Val  RMSE                : {r['final_val_rmse']:.4f} kW")
print(f"  Val  MAE                 : {r['final_val_mae']:.4f} kW")
print(f"  Test NRMSE (Apr+Sep 2025): {nrmse_test:.3f}%")
print(f"  Test RMSE                : {rmse_test:.4f} kW")
print(f"  Test MAE                 : {mae_test:.4f} kW")
print(f"  Predictions saved to {pred_dir}")
