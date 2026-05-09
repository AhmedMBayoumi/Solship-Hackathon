"""
With walkforward training (2024 + 2025 up to test month), the 2024->2025
distribution gap is much smaller. Re-test if lighter regularization now helps.

Compare: heavy / medium / light reg for both v2 and v4 features.
Also test 24-bag ensemble for the best config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import time

ROOT = Path(__file__).parents[1]

DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
             "qow","hod","net_load","sell_price"}

REG_CONFIGS = {
    "heavy": dict(num_leaves=15, max_depth=4, min_child_samples=100,
                  reg_alpha=2.0, reg_lambda=3.0, subsample=0.7, colsample_bytree=0.7),
    "medium":dict(num_leaves=31, max_depth=6, min_child_samples=50,
                  reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8),
    "light": dict(num_leaves=63, max_depth=8, min_child_samples=20,
                  reg_alpha=0.1, reg_lambda=0.1, subsample=0.9, colsample_bytree=0.9),
}

def april_split(df):
    ts = df["timestamp"]
    return (((ts.dt.year == 2024) | ((ts.dt.year == 2025) & (ts.dt.month <= 2))),
            ((ts.dt.year == 2025) & (ts.dt.month == 3)),
            ((ts.dt.year == 2025) & (ts.dt.month == 4)))

def sept_split(df):
    ts = df["timestamp"]
    return (((ts.dt.year == 2024) |
             ((ts.dt.year == 2025) & (ts.dt.month <= 7)) |
             ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day <= 15))),
            ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day > 15)),
            ((ts.dt.year == 2025) & (ts.dt.month == 9)))

def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)
def mae(y, yp):   return float(np.mean(np.abs(y - yp)))


def train_one(df_all, feats, splits, reg_label, n_bags=12):
    tr_m, va_m, te_m = splits
    train_df = df_all[tr_m]; val_df = df_all[va_m]; test_df = df_all[te_m]
    tv_df = pd.concat([train_df, val_df], ignore_index=True)
    X_va, y_va = val_df[feats].values,  val_df["load_kw"].values
    X_te, y_te = test_df[feats].values, test_df["load_kw"].values
    X_tv, y_tv = tv_df[feats].values,   tv_df["load_kw"].values

    base = REG_CONFIGS[reg_label]
    val_preds = np.zeros((n_bags, len(y_va)))
    test_preds= np.zeros((n_bags, len(y_te)))
    for i in range(n_bags):
        seed = 42 + i
        p = dict(base)
        p.update({"n_estimators": 3000, "learning_rate": 0.01, "subsample_freq": 1,
                  "objective": "huber", "alpha": 0.9,
                  "verbose": -1, "n_jobs": -1, "random_state": seed})
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_tv), size=int(len(X_tv) * 0.9))
        m = lgb.LGBMRegressor(**p); m.fit(X_tv[idx], y_tv[idx])
        val_preds[i]  = np.clip(m.predict(X_va), 0, None)
        test_preds[i] = np.clip(m.predict(X_te), 0, None)
    return val_preds.mean(0), test_preds.mean(0), y_va, y_te, test_df["timestamp"].values


def run_for_features(feature_version, n_bags=8):
    df_all = pd.read_parquet(ROOT / f"data/features/features_{feature_version}_all.parquet")
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    feats = [c for c in df_all.columns if c not in DROP_BASE]
    print(f"\n{'='*70}")
    print(f"Features: {feature_version} ({len(feats)} features)")
    print(f"{'='*70}")
    for reg in ["heavy", "medium", "light"]:
        t0 = time.time()
        # April
        _, p_a, _, y_a, ts_a = train_one(df_all, feats, april_split(df_all), reg, n_bags)
        # September
        _, p_s, _, y_s, ts_s = train_one(df_all, feats, sept_split(df_all), reg, n_bags)
        y_all = np.concatenate([y_a, y_s])
        p_all = np.concatenate([p_a, p_s])
        n_a = nrmse(y_a, p_a); n_s = nrmse(y_s, p_s); n_c = nrmse(y_all, p_all)
        ma  = mae(y_all, p_all)
        print(f"  reg={reg:<6s}  bags={n_bags}   April {n_a:.2f}%  Sept {n_s:.2f}%  Combined {n_c:.2f}%  MAE={ma:.4f}  ({time.time()-t0:.0f}s)")
        # Save best
        out_path = ROOT / f"outputs/forecasts/bagging_{feature_version}_{reg}_test_preds.csv"
        ts_all = np.concatenate([ts_a, ts_s])
        pd.DataFrame({"timestamp": ts_all, "load_pred": p_all}).to_csv(out_path, index=False)


# Run for v2 and v4
run_for_features("v2", n_bags=8)
run_for_features("v4", n_bags=8)

print("\nDone. Compare CSVs in outputs/forecasts/")
