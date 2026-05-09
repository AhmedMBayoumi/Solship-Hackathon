"""Check importance of NEW v3 features vs v2 features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
df_all = pd.read_parquet(ROOT / "data/features/features_v3_all.parquet")
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
             "qow","hod","net_load","sell_price"}
feats = [c for c in df_all.columns if c not in DROP_BASE]

# v2 features for comparison
v2_feats = set((ROOT / "data/features/features_v2_cols.txt").read_text().splitlines())

ts = df_all["timestamp"]
train_m = ((ts.dt.year == 2024) | ((ts.dt.year == 2025) & (ts.dt.month <= 7)))
val_m   = ((ts.dt.year == 2025) & (ts.dt.month == 8))
df_tr = df_all[train_m].dropna(subset=feats)
df_va = df_all[val_m].dropna(subset=feats)

X_tr, y_tr = df_tr[feats].values, df_tr["load_kw"].values
X_va, y_va = df_va[feats].values, df_va["load_kw"].values

print(f"Training LGBM on Sept-window train data ({len(X_tr)} rows, {len(feats)} feats)...")
m = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.01, num_leaves=15, max_depth=4,
    min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    objective="huber", alpha=0.9,
    verbose=-1, n_jobs=-1, random_state=42,
)
m.fit(X_tr, y_tr)

imp = m.feature_importances_
imp_df = pd.DataFrame({"feature": feats, "importance": imp,
                       "is_new_v3": [f not in v2_feats for f in feats]})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

# Top 30 overall
print("\nTOP 30 FEATURES OVERALL (★ = NEW in v3):")
for i, r in imp_df.head(30).iterrows():
    star = "★" if r["is_new_v3"] else " "
    print(f"  {star} {r['feature']:30s}  {r['importance']}")

# v3-only ranking
print(f"\nNEW v3 FEATURES RANKED (top 25):")
v3_only = imp_df[imp_df["is_new_v3"]].sort_values("importance", ascending=False).head(25)
for i, r in v3_only.iterrows():
    print(f"  {r['feature']:30s}  importance={r['importance']:>5}  rank={i+1}/{len(imp_df)}")

# Statistics
total_v2 = imp_df[~imp_df["is_new_v3"]]["importance"].sum()
total_v3_new = imp_df[imp_df["is_new_v3"]]["importance"].sum()
print(f"\nIMPORTANCE BUDGET:")
print(f"  v2 features (88): total importance = {total_v2:>8}  ({total_v2/(total_v2+total_v3_new):.1%})")
print(f"  v3 NEW (43)     : total importance = {total_v3_new:>8}  ({total_v3_new/(total_v2+total_v3_new):.1%})")

# Useless features (importance < 50)
useless = imp_df[imp_df["importance"] < 50]
print(f"\n  Useless features (importance < 50): {len(useless)}")
for _, r in useless.iterrows():
    new_tag = " (NEW v3)" if r["is_new_v3"] else ""
    print(f"    {r['feature']:30s}  imp={r['importance']:>4}{new_tag}")

# Save full ranking
imp_df.to_csv(ROOT / "outputs/v3_feature_importance.csv", index=False)
print(f"\nSaved -> outputs/v3_feature_importance.csv")
