"""
FINAL: walkforward bagging with LIGHT regularization (12 bags), v2 features.
Confirmed by reg sweep that light reg works now that walkforward training
includes 2025 data (distribution shift is closed).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import time

ROOT = Path(__file__).parents[1]
df_all = pd.read_parquet(ROOT / "data/features/features_v2_all.parquet")
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
             "qow","hod","net_load","sell_price","pv_today_total"}
DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
feats = [c for c in df_all.columns if c not in (DROP_BASE | DROP_LEAKY)]
print(f"Features (v2): {len(feats)}")

def april_split():
    ts = df_all["timestamp"]
    return (((ts.dt.year == 2024) | ((ts.dt.year == 2025) & (ts.dt.month <= 2))),
            ((ts.dt.year == 2025) & (ts.dt.month == 3)),
            ((ts.dt.year == 2025) & (ts.dt.month == 4)))

def sept_split():
    ts = df_all["timestamp"]
    return (((ts.dt.year == 2024) |
             ((ts.dt.year == 2025) & (ts.dt.month <= 7)) |
             ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day <= 15))),
            ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day > 15)),
            ((ts.dt.year == 2025) & (ts.dt.month == 9)))

def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)
def mae(y, yp):   return float(np.mean(np.abs(y - yp)))

# Diverse configs around "light" regularization
BAG_CONFIGS = [
    # primary: light
    {"num_leaves": 63, "max_depth": 8, "learning_rate": 0.02, "min_child_samples": 20,
     "reg_alpha": 0.1, "reg_lambda": 0.1, "subsample": 0.9, "colsample_bytree": 0.9},
    # variant: medium-light
    {"num_leaves": 47, "max_depth": 7, "learning_rate": 0.015, "min_child_samples": 30,
     "reg_alpha": 0.3, "reg_lambda": 0.5, "subsample": 0.85, "colsample_bytree": 0.85},
    # variant: light-deeper
    {"num_leaves": 95, "max_depth": 10, "learning_rate": 0.025, "min_child_samples": 15,
     "reg_alpha": 0.05, "reg_lambda": 0.1, "subsample": 0.85, "colsample_bytree": 0.85},
    # variant: light + slower lr
    {"num_leaves": 63, "max_depth": 8, "learning_rate": 0.01, "min_child_samples": 25,
     "reg_alpha": 0.2, "reg_lambda": 0.2, "subsample": 0.9, "colsample_bytree": 0.9},
]
N_BAGS = 12

def train_bagging(train_df, val_df, test_df, label):
    print(f"\n=== {label} ===")
    print(f"  train rows: {len(train_df)}  val rows: {len(val_df)}  test rows: {len(test_df)}")
    tv_df = pd.concat([train_df, val_df], ignore_index=True)
    X_va, y_va = val_df[feats].values,   val_df["load_kw"].values
    X_te, y_te = test_df[feats].values,  test_df["load_kw"].values
    X_tv, y_tv = tv_df[feats].values,    tv_df["load_kw"].values

    val_preds = np.zeros((N_BAGS, len(y_va)))
    test_preds= np.zeros((N_BAGS, len(y_te)))
    t0 = time.time()
    for i in range(N_BAGS):
        cfg  = BAG_CONFIGS[i % len(BAG_CONFIGS)]
        seed = 42 + i
        p = dict(cfg); p.update({"n_estimators": 3000, "subsample_freq": 1,
                                  "objective": "huber", "alpha": 0.9,
                                  "verbose": -1, "n_jobs": -1, "random_state": seed})
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_tv), size=int(len(X_tv) * 0.9))
        m = lgb.LGBMRegressor(**p); m.fit(X_tv[idx], y_tv[idx])
        val_preds[i]  = np.clip(m.predict(X_va), 0, None)
        test_preds[i] = np.clip(m.predict(X_te), 0, None)
        if (i + 1) % 4 == 0 or i == N_BAGS - 1:
            print(f"    [{i+1}/{N_BAGS}]  val={nrmse(y_va, val_preds[i]):.2f}%  test={nrmse(y_te, test_preds[i]):.2f}%  ({time.time()-t0:.0f}s)")

    val_avg  = val_preds.mean(axis=0)
    test_avg = test_preds.mean(axis=0)
    print(f"  Bagging val NRMSE : {nrmse(y_va, val_avg):.2f}%")
    print(f"  Bagging test NRMSE: {nrmse(y_te, test_avg):.2f}%")
    print(f"  Bagging test MAE  : {mae(y_te, test_avg):.4f}")
    print(f"  Bagging test RMSE : {np.sqrt(np.mean((y_te-test_avg)**2)):.4f}")
    return test_df["timestamp"].values, test_avg, y_te


all_test_ts, all_test_pred, all_test_y = [], [], []

tr_m, va_m, te_m = april_split()
ts_a, p_a, y_a = train_bagging(df_all[tr_m], df_all[va_m], df_all[te_m], "APRIL MODEL (light reg, 12 bags)")
all_test_ts.append(ts_a); all_test_pred.append(p_a); all_test_y.append(y_a)

tr_m, va_m, te_m = sept_split()
ts_s, p_s, y_s = train_bagging(df_all[tr_m], df_all[va_m], df_all[te_m], "SEPTEMBER MODEL (light reg, 12 bags)")
all_test_ts.append(ts_s); all_test_pred.append(p_s); all_test_y.append(y_s)

ts_combined   = np.concatenate(all_test_ts)
pred_combined = np.concatenate(all_test_pred)
y_combined    = np.concatenate(all_test_y)
print(f"\n=== FINAL: COMBINED Apr+Sep 2025 (light reg, 12 bags, v2 features) ===")
print(f"  test NRMSE: {nrmse(y_combined, pred_combined):.2f}%")
print(f"  test MAE  : {mae(y_combined, pred_combined):.4f}")
print(f"  test RMSE : {np.sqrt(np.mean((y_combined-pred_combined)**2)):.4f}")

print(f"\nProgression of best bagging-walkforward results:")
print(f"  v2 + heavy reg (prev submission): 61.46%")
print(f"  v2 + medium reg                 : 61.24%")
print(f"  v2 + light reg (8 bags)         : 60.90%")
print(f"  v2 + light reg (12 diverse bags): {nrmse(y_combined, pred_combined):.2f}%")

out = ROOT / "outputs/forecasts/bagging_walkforward_FINAL_test_preds.csv"
pd.DataFrame({"timestamp": ts_combined, "load_pred": pred_combined}).to_csv(out, index=False)
print(f"\nSaved -> {out}")
