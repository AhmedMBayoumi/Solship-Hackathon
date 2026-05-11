"""LOCAL: online retraining LGBM on surprise features. Predict March 2026."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import time

ROOT = Path(__file__).parents[1]
FEAT = ROOT / "data/features/features_surprise_all.parquet"

def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))
def mae(y, yp):  return float(np.mean(np.abs(y-yp)))
def mape(y, yp): return float(np.mean(np.abs(y-yp) / np.maximum(np.abs(y), 0.01)) * 100)

def main():
    print(f"Loading {FEAT.name}...")
    df = pd.read_parquet(FEAT)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True).ffill().bfill()
    print(f"  rows={len(df)}  cols={df.shape[1]}")

    DROP_COLS = ["timestamp", "load_kw", "pv_kw", "minute", "qow", "hod", "net_load"]
    feats = [c for c in df.columns if c not in DROP_COLS]
    print(f"  features: {len(feats)}")

    # Test = March 2026 (last month). Retrain every 3 days.
    test_mask  = (df["timestamp"].dt.year == 2026) & (df["timestamp"].dt.month == 3)
    test_days  = sorted(df[test_mask]["timestamp"].dt.day.unique())
    retrain_freq = 3

    all_preds, all_actuals, all_ts = [], [], []
    t0 = time.time()
    for i in range(0, len(test_days), retrain_freq):
        current_day = test_days[i]
        next_days   = test_days[i:i+retrain_freq]
        train_mask  = df["timestamp"] < pd.Timestamp(2026, 3, current_day)
        sub_test    = test_mask & df["timestamp"].dt.day.isin(next_days)

        X_tr = df[train_mask][feats]
        y_tr = df[train_mask]["load_kw"]
        X_te = df[sub_test][feats]
        y_te = df[sub_test]["load_kw"]
        ts_te = df[sub_test]["timestamp"]
        if len(X_te) == 0: continue
        print(f"  Day {current_day:2d}: train={len(X_tr)}  test={len(X_te)}  ({time.time()-t0:.0f}s)")

        m = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.04, num_leaves=31,
            reg_alpha=0.2, reg_lambda=0.2, objective="huber",
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        p = np.clip(m.predict(X_te), 0, None)

        # Bias correction: last 2 days residual
        recent = (df["timestamp"] < pd.Timestamp(2026, 3, current_day)) & \
                 (df["timestamp"] >= pd.Timestamp(2026, 3, current_day) - pd.Timedelta(days=2))
        if recent.sum() > 0:
            bias = df[recent]["load_kw"].mean() - m.predict(df[recent][feats]).mean()
            p = np.clip(p + bias, 0, None)

        all_preds.extend(p)
        all_actuals.extend(y_te)
        all_ts.extend(ts_te)

    y_all = np.array(all_actuals)
    p_all = np.array(all_preds)
    print(f"\n=== ONLINE RETRAINING (surprise / March 2026) ===")
    print(f"  RMSE  : {rmse(y_all, p_all):.4f} kW")
    print(f"  MAE   : {mae(y_all, p_all):.4f} kW")
    print(f"  MAPE  : {mape(y_all, p_all):.2f} %")
    print(f"  NRMSE : {nrmse(y_all, p_all):.2f} %")

    out = ROOT / "outputs/forecasts/surprise_online_retraining_test_preds.csv"
    pd.DataFrame({"timestamp": all_ts, "load_pred": p_all}).to_csv(out, index=False)
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()
