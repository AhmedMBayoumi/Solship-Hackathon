"""
Walk-forward training: per-month-of-test models that include all data up
to the test month (per supervisor's allowance).

April model    : train = 2024 + 2025 Jan-Feb,        val = 2025 Mar
September model: train = 2024 + 2025 Jan-Aug excl Sep, val = 2025 Aug
                 (uses 2025 Apr actuals too — they're <Aug 2025, allowed)

Bagging: 12× LGBM with diverse hyperparams + bootstrap.
Per supervisor: horizon for MPC will be capped at H<=96 (24h, battery cycle).
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

DROP_BASE  = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
              "qow","hod","net_load","sell_price","pv_today_total"}
DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
feats = [c for c in df_all.columns if c not in (DROP_BASE | DROP_LEAKY)]
print(f"Features: {len(feats)}")

# Build the two splits
def april_split():
    """April 2025 model: train = 2024 + 2025 Jan-Feb,  val = 2025 Mar,  test = 2025 Apr."""
    ts = df_all["timestamp"]
    train_m = ((ts.dt.year == 2024) |
               ((ts.dt.year == 2025) & (ts.dt.month <= 2)))
    val_m   = ((ts.dt.year == 2025) & (ts.dt.month == 3))
    test_m  = ((ts.dt.year == 2025) & (ts.dt.month == 4))
    return train_m, val_m, test_m

def sept_split():
    """September 2025 model: train = 2024 + 2025 Jan-Aug, val = 2025 Aug, test = 2025 Sep.
    To keep val a held-out window, we use 2025 Aug as val and put 2025 Jul into train.
    But actually use the last 4 weeks of Aug as val so models still see Apr-Jul 2025."""
    ts = df_all["timestamp"]
    # Train: 2024 + 2025 Jan-Jul + first half of Aug
    train_m = ((ts.dt.year == 2024) |
               ((ts.dt.year == 2025) & (ts.dt.month <= 7)) |
               ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day <= 15)))
    val_m   = ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day > 15))
    test_m  = ((ts.dt.year == 2025) & (ts.dt.month == 9))
    return train_m, val_m, test_m

def nrmse(y, yp):
    return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100) if np.mean(y) > 0 else float("nan")
def mae(y, yp):
    return float(np.mean(np.abs(y - yp)))

# Heavy-reg + huber baseline + bagging configs (best so far)
BAG_CONFIGS = [
    {"num_leaves": 15, "max_depth": 4,  "learning_rate": 0.01, "min_child_samples": 100, "reg_alpha": 2.0, "reg_lambda": 3.0, "subsample": 0.7, "colsample_bytree": 0.7},
    {"num_leaves": 31, "max_depth": 5,  "learning_rate": 0.02, "min_child_samples": 50,  "reg_alpha": 1.0, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"num_leaves": 7,  "max_depth": 3,  "learning_rate": 0.005,"min_child_samples": 200, "reg_alpha": 5.0, "reg_lambda": 5.0, "subsample": 0.6, "colsample_bytree": 0.6},
    {"num_leaves": 63, "max_depth": 6,  "learning_rate": 0.03, "min_child_samples": 30,  "reg_alpha": 0.5, "reg_lambda": 0.5, "subsample": 0.9, "colsample_bytree": 0.9},
]
N_BAGS = 12

def train_bagging(train_df, val_df, test_df, label):
    print(f"\n=== {label} ===")
    print(f"  train rows: {len(train_df)}  val rows: {len(val_df)}  test rows: {len(test_df)}")
    print(f"  train time range: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    print(f"  val   time range: {val_df['timestamp'].min()} -> {val_df['timestamp'].max()}")
    print(f"  test  time range: {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")

    # combine train + val for final fit (val NRMSE is for sanity only)
    tv_df = pd.concat([train_df, val_df], ignore_index=True)
    X_tr, y_tr = train_df[feats].values, train_df["load_kw"].values
    X_va, y_va = val_df[feats].values,   val_df["load_kw"].values
    X_te, y_te = test_df[feats].values,  test_df["load_kw"].values
    X_tv, y_tv = tv_df[feats].values,    tv_df["load_kw"].values

    val_preds  = np.zeros((N_BAGS, len(y_va)))
    test_preds = np.zeros((N_BAGS, len(y_te)))
    t0 = time.time()
    for i in range(N_BAGS):
        cfg  = BAG_CONFIGS[i % len(BAG_CONFIGS)]
        seed = 42 + i
        p = dict(cfg)
        p.update({"n_estimators": 3000, "subsample_freq": 1,
                  "objective": "huber", "alpha": 0.9,
                  "verbose": -1, "n_jobs": -1, "random_state": seed})
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_tv), size=int(len(X_tv) * 0.9))
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tv[idx], y_tv[idx])
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
    return test_df["timestamp"].values, test_avg, val_avg, y_te, y_va


# ── Train both models ─────────────────────────────────────────
all_test_ts   = []
all_test_pred = []
all_test_y    = []

# April
tr_m, va_m, te_m = april_split()
ts_a, p_a, _, y_a, _ = train_bagging(df_all[tr_m], df_all[va_m], df_all[te_m], "APRIL MODEL")
all_test_ts  .append(ts_a); all_test_pred.append(p_a); all_test_y.append(y_a)

# September
tr_m, va_m, te_m = sept_split()
ts_s, p_s, _, y_s, _ = train_bagging(df_all[tr_m], df_all[va_m], df_all[te_m], "SEPTEMBER MODEL")
all_test_ts  .append(ts_s); all_test_pred.append(p_s); all_test_y.append(y_s)

# Combine into one prediction file for MPC
ts_combined   = np.concatenate(all_test_ts)
pred_combined = np.concatenate(all_test_pred)
y_combined    = np.concatenate(all_test_y)
print(f"\n=== Combined April + September test set ===")
print(f"  Total rows: {len(ts_combined)}")
print(f"  test NRMSE: {nrmse(y_combined, pred_combined):.2f}%")
print(f"  test MAE  : {mae(y_combined, pred_combined):.4f}")
print(f"  test RMSE : {np.sqrt(np.mean((y_combined-pred_combined)**2)):.4f}")

out = ROOT / "outputs/forecasts/bagging_walkforward_test_preds.csv"
pd.DataFrame({"timestamp": ts_combined, "load_pred": pred_combined}).to_csv(out, index=False)
print(f"\nSaved -> {out}")
