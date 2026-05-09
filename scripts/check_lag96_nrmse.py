"""Check lag_96 persistence NRMSE on Apr+Sep 2025 + load distribution shift."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parents[1]
df = pd.read_parquet(ROOT / "data/features/features_all.parquet")

# 2025 test slice
df_25 = df[df["timestamp"].dt.year == 2025].sort_values("timestamp").reset_index(drop=True)
test = df_25[df_25["timestamp"].dt.month.isin([4, 9])].copy()

y  = test["load_kw"].values
yp = test["lag_96"].values
mask = ~np.isnan(yp)
y, yp = y[mask], yp[mask]
rmse = np.sqrt(np.mean((y - yp) ** 2))
print(f"lag_96 persistence on Apr+Sep 2025:")
print(f"  N={len(y)}  RMSE={rmse:.4f}  NRMSE={rmse/y.mean()*100:.2f}%  mean(load)={y.mean():.4f}")

for mo, name in [(4, "April"), (9, "September")]:
    m = test[test["timestamp"].dt.month == mo].dropna(subset=["lag_96"])
    yt, yp_m = m["load_kw"].values, m["lag_96"].values
    rmse_m = np.sqrt(np.mean((yt - yp_m) ** 2))
    print(f"  {name}: NRMSE={rmse_m/yt.mean()*100:.2f}%  RMSE={rmse_m:.4f}  mean={yt.mean():.4f}")

print()
print("Load distribution 2024 vs 2025:")
df_24 = df[df["timestamp"].dt.year == 2024]
print(f"  2024 mean load: {df_24['load_kw'].mean():.4f}  median: {df_24['load_kw'].median():.4f}")
print(f"  2025 mean load: {df_25['load_kw'].mean():.4f}  median: {df_25['load_kw'].median():.4f}")

m24 = df_24[df_24["timestamp"].dt.month.isin([4, 9])]
m25 = df_25[df_25["timestamp"].dt.month.isin([4, 9])]
print(f"  2024 Apr+Sep mean: {m24['load_kw'].mean():.4f}")
print(f"  2025 Apr+Sep mean: {m25['load_kw'].mean():.4f}")

# Also: lag_192 (2 days), lag_672 (1 week)
for col in ["lag_192", "lag_672"]:
    sub = test.dropna(subset=[col])
    rmse_c = np.sqrt(np.mean((sub["load_kw"] - sub[col]) ** 2))
    print(f"  {col} NRMSE: {rmse_c/sub['load_kw'].mean()*100:.2f}%")
