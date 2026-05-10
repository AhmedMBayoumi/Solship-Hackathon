"""
scripts/online_retraining_lgbm.py
Aggressive online retraining: every week in 2025, we retrain the model
on all ground truth data available up to that week.
This is the highest-fidelity interpretation of "you have all previous data".
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import time

ROOT = Path(__file__).parents[1]
FEAT_PATH = ROOT / "data/features/features_v5_all.parquet"

def nrmse(y, yp):
    return float(np.sqrt(np.mean((y - yp)**2)) / np.mean(y) * 100)

def main():
    print("Loading features...")
    df = pd.read_parquet(FEAT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.ffill().bfill()

    DROP_COLS = ["timestamp", "load_kw", "p_battery_kw", "grid_kw", 
                 "tariff_band", "net_load", "sell_price"]
    feats = [c for c in df.columns if c not in DROP_COLS]
    
    # Target months: April (4) and September (9)
    test_months = [4, 9]
    all_preds = []
    all_actuals = []

    for month in test_months:
        print(f"\n--- Processing Month {month} ---")
        month_mask = (df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month == month)
        month_days = df[month_mask]["timestamp"].dt.day.unique()
        
        # We will retrain every 3 days to balance speed and accuracy
        retrain_freq = 3 
        
        for i in range(0, len(month_days), retrain_freq):
            current_day = month_days[i]
            # Data available: Everything before this day
            train_mask = (df["timestamp"] < pd.Timestamp(2025, month, current_day))
            # Test data: Next 3 days
            next_3_days = month_days[i : i+retrain_freq]
            test_mask = (df["timestamp"].dt.year == 2025) & \
                        (df["timestamp"].dt.month == month) & \
                        (df["timestamp"].dt.day.isin(next_3_days))
            
            X_tr, y_tr = df[train_mask][feats], df[train_mask]["load_kw"]
            X_te, y_te = df[test_mask][feats],  df[test_mask]["load_kw"]
            
            if len(X_te) == 0: continue
            
            print(f"  Day {current_day:2d}: Training on {len(X_tr)} samples, predicting {len(X_te)}...")
            
            model = lgb.LGBMRegressor(
                n_estimators=1500,
                learning_rate=0.04,
                num_leaves=31,
                reg_alpha=0.2,
                reg_lambda=0.2,
                objective="huber",
                random_state=42,
                verbose=-1
            )
            model.fit(X_tr, y_tr)
            p = np.clip(model.predict(X_te), 0, None)
            
            # Local bias correction based on last 2 days of training data
            # (since we are in the same year/month now, this is very effective)
            recent_train_mask = (df["timestamp"] < pd.Timestamp(2025, month, current_day)) & \
                                (df["timestamp"] >= pd.Timestamp(2025, month, current_day) - pd.Timedelta(days=2))
            if recent_train_mask.sum() > 0:
                recent_actual = df[recent_train_mask]["load_kw"].mean()
                recent_pred   = model.predict(df[recent_train_mask][feats]).mean()
                bias = recent_actual - recent_pred
                p = np.clip(p + bias, 0, None)
            
            all_preds.extend(p)
            all_actuals.extend(y_te)

    y_all = np.array(all_actuals)
    p_all = np.array(all_preds)
    
    score = nrmse(y_all, p_all)
    print(f"\nFINAL ONLINE RETRAINING NRMSE: {score:.2f}%")
    
    # Save
    out = ROOT / "outputs/forecasts/online_retraining_test_preds.csv"
    pd.DataFrame({
        "timestamp": df[(df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month.isin([4, 9]))]["timestamp"],
        "load_pred": p_all
    }).to_csv(out, index=False)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
