"""
Analyze forecast bias vs actual on Apr+Sep 2025 test, and try
a bias-corrected version + a calendar-mean baseline.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

# ── Load actuals + predictions ────────────────────────────────────────
df_test = pd.read_parquet(ROOT / "data/features/features_test.parquet")
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
df_test = df_test.sort_values("timestamp").reset_index(drop=True)

lgbm = pd.read_csv(ROOT / "outputs/forecasts/lgbm_test_preds.csv", parse_dates=["timestamp"])
xgb  = pd.read_csv(ROOT / "outputs/forecasts/xgb_test_preds.csv",  parse_dates=["timestamp"])

m = df_test[["timestamp", "load_kw"]].merge(lgbm.rename(columns={"load_pred":"lgbm"}), on="timestamp", how="left")
m = m.merge(xgb.rename(columns={"load_pred":"xgb"}), on="timestamp", how="left")

print("Forecast statistics on Apr+Sep 2025:")
print(f"  actual    mean={m['load_kw'].mean():.4f}  std={m['load_kw'].std():.4f}")
print(f"  lgbm      mean={m['lgbm'].mean():.4f}  std={m['lgbm'].std():.4f}")
print(f"  xgb       mean={m['xgb'].mean():.4f}  std={m['xgb'].std():.4f}")
print(f"  bias_lgbm = mean(pred - actual) = {(m['lgbm'] - m['load_kw']).mean():+.4f}")
print(f"  bias_xgb  = mean(pred - actual) = {(m['xgb']  - m['load_kw']).mean():+.4f}")

# ── Bias correction (using val 2024 bias as proxy) ────────────────────
lgbm_val = pd.read_csv(ROOT / "outputs/forecasts/lgbm_val_preds.csv", parse_dates=["timestamp"])
xgb_val  = pd.read_csv(ROOT / "outputs/forecasts/xgb_val_preds.csv",  parse_dates=["timestamp"])
df_val   = pd.read_parquet(ROOT / "data/features/features_val.parquet")
df_val["timestamp"] = pd.to_datetime(df_val["timestamp"])
mv = df_val[["timestamp", "load_kw"]].merge(lgbm_val.rename(columns={"load_pred":"lgbm"}), on="timestamp", how="left")
mv = mv.merge(xgb_val.rename(columns={"load_pred":"xgb"}), on="timestamp", how="left")

bias_lgbm_val = (mv["lgbm"] - mv["load_kw"]).mean()
bias_xgb_val  = (mv["xgb"]  - mv["load_kw"]).mean()
ratio_lgbm_val = mv["load_kw"].mean() / mv["lgbm"].mean()
ratio_xgb_val  = mv["load_kw"].mean() / mv["xgb"].mean()

print()
print("Validation 2024 bias:")
print(f"  bias_lgbm_val = {bias_lgbm_val:+.4f}  ratio={ratio_lgbm_val:.4f}")
print(f"  bias_xgb_val  = {bias_xgb_val:+.4f}  ratio={ratio_xgb_val:.4f}")

# Apply both corrections to test and recompute NRMSE
def nrmse(y, yp):
    return np.sqrt(np.mean((y - yp) ** 2)) / y.mean() * 100

print()
print("Test NRMSE with various corrections:")
for name, col, bias, ratio in [
    ("LGBM raw", "lgbm", 0.0, 1.0),
    ("LGBM bias-sub", "lgbm", bias_lgbm_val, 1.0),
    ("LGBM ratio", "lgbm", 0.0, ratio_lgbm_val),
    ("XGB raw", "xgb", 0.0, 1.0),
    ("XGB bias-sub", "xgb", bias_xgb_val, 1.0),
    ("XGB ratio", "xgb", 0.0, ratio_xgb_val),
]:
    pred = (m[col] - bias) * ratio
    pred = pred.clip(lower=0)
    n = nrmse(m["load_kw"], pred)
    print(f"  {name:18s} mean(pred)={pred.mean():.4f}  NRMSE={n:.2f}%")

# Ensemble (average of LGBM and XGB)
m["ens_raw"]  = 0.5 * m["lgbm"] + 0.5 * m["xgb"]
m["ens_corr"] = 0.5 * (m["lgbm"] - bias_lgbm_val) + 0.5 * (m["xgb"] - bias_xgb_val)
print()
print(f"  Ensemble raw       mean={m['ens_raw'].mean():.4f}   NRMSE={nrmse(m['load_kw'], m['ens_raw']):.2f}%")
print(f"  Ensemble bias-corr mean={m['ens_corr'].mean():.4f}   NRMSE={nrmse(m['load_kw'], m['ens_corr'].clip(lower=0)):.2f}%")

# ── Calendar-mean baseline (hour × dow from 2024) ─────────────────────
df_train = pd.read_parquet(ROOT / "data/features/features_train.parquet")
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
df_trainval = pd.concat([df_train, df_val], ignore_index=True)
# Hour × dow mean
hod_mean = df_trainval.groupby([df_trainval["timestamp"].dt.dayofweek,
                                df_trainval["timestamp"].dt.hour,
                                df_trainval["timestamp"].dt.minute])["load_kw"].mean()
# Build calendar-mean prediction for test
m["dow"]    = m["timestamp"].dt.dayofweek
m["hour"]   = m["timestamp"].dt.hour
m["minute"] = m["timestamp"].dt.minute
m["cal_mean"] = m.apply(lambda r: hod_mean.get((r["dow"], r["hour"], r["minute"]), 0.9), axis=1)
print(f"  Cal-mean (dow×h×m) mean={m['cal_mean'].mean():.4f}   NRMSE={nrmse(m['load_kw'], m['cal_mean']):.2f}%")

# Save bias-corrected predictions for MPC use
out_dir = ROOT / "outputs/forecasts"
m_save = m[["timestamp"]].copy()
m_save["load_pred"] = (m["lgbm"] - bias_lgbm_val).clip(lower=0)
m_save.to_csv(out_dir / "lgbm_biascorr_test_preds.csv", index=False)
print(f"\nSaved bias-corrected LGBM preds -> {out_dir/'lgbm_biascorr_test_preds.csv'}")

m_save["load_pred"] = m["ens_corr"].clip(lower=0)
m_save.to_csv(out_dir / "ensemble_test_preds.csv", index=False)
print(f"Saved ensemble preds -> {out_dir/'ensemble_test_preds.csv'}")

m_save["load_pred"] = m["cal_mean"]
m_save.to_csv(out_dir / "cal_mean_test_preds.csv", index=False)
print(f"Saved cal-mean preds -> {out_dir/'cal_mean_test_preds.csv'}")
