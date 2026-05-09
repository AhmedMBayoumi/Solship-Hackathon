"""
Estimate how much variance is INHERENTLY UNPREDICTABLE for residential load.
Tells us the achievable NRMSE floor.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
df = pd.read_parquet(ROOT / "data/features/features_all.parquet")
df = df.sort_values("timestamp").reset_index(drop=True)

# 2024 Apr+Sep (val period)
val24 = df[(df["timestamp"].dt.year == 2024) & (df["timestamp"].dt.month.isin([4,9]))].copy()
# 2025 Apr+Sep (test period)
te25  = df[(df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month.isin([4,9]))].copy()

def nrmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)) / y.mean() * 100)


print("===== NOISE FLOOR ANALYSIS =====\n")
for label, d in [("2024 Apr+Sep (val)", val24), ("2025 Apr+Sep (test)", te25)]:
    y = d["load_kw"].values
    print(f"\n{label}:")
    print(f"  N            : {len(y)}")
    print(f"  mean(load)   : {y.mean():.4f} kW")
    print(f"  std(load)    : {y.std():.4f}")
    print(f"  CV = std/mean: {y.std()/y.mean():.4f}")
    # ACF
    for lag in [1, 4, 96, 672]:
        if len(y) > lag:
            acf = np.corrcoef(y[lag:], y[:-lag])[0, 1]
            print(f"  ACF lag-{lag:>3}: {acf:.4f}")

    # Persistence baselines
    sub = d.dropna(subset=["lag_1", "lag_96", "lag_672"])
    yt = sub["load_kw"].values
    print(f"  --- Persistence baselines ---")
    print(f"  mean predictor      NRMSE: {nrmse(yt, np.full_like(yt, yt.mean())):>6.2f}%")
    print(f"  lag_1               NRMSE: {nrmse(yt, sub['lag_1'].values):>6.2f}%")
    print(f"  lag_96              NRMSE: {nrmse(yt, sub['lag_96'].values):>6.2f}%")
    print(f"  lag_672             NRMSE: {nrmse(yt, sub['lag_672'].values):>6.2f}%")
    # Best of recent lags
    best = np.minimum(np.minimum(np.abs(yt - sub['lag_96']), np.abs(yt - sub['lag_672'])), np.abs(yt - sub['lag_1']))
    print(f"  best of {{1,96,672}} (oracle ensemble) NRMSE: {nrmse(yt, yt + np.where(np.abs(yt-sub['lag_96'])<np.abs(yt-sub['lag_672']), sub['lag_96']-yt, sub['lag_672']-yt)):>6.2f}%")
    # Average of lag_96 and lag_672
    avg = 0.5*(sub['lag_96'].values + sub['lag_672'].values)
    print(f"  avg(lag_96, lag_672) NRMSE: {nrmse(yt, avg):>6.2f}%")

    # Hour-of-week mean from same year
    sub2 = d.copy()
    sub2["hod"] = sub2["timestamp"].dt.hour
    sub2["dow"] = sub2["timestamp"].dt.dayofweek
    grp = sub2.groupby(["dow", "hod"])["load_kw"].mean()
    sub2["how_mean"] = sub2.apply(lambda r: grp[(r["dow"], r["hod"])], axis=1)
    print(f"  hour-of-week mean   NRMSE: {nrmse(sub2['load_kw'], sub2['how_mean']):>6.2f}%")

# Distribution shift check
print("\n===== DISTRIBUTION SHIFT 2024 -> 2025 =====")
for col in ["load_kw", "pv_kw", "temperature_2m", "shortwave_radiation"]:
    if col not in df.columns:
        continue
    a = val24[col].mean()
    b = te25[col].mean()
    print(f"  {col:>22}: 2024={a:.3f}  2025={b:.3f}  delta={b-a:+.3f}  ({(b-a)/a*100:+.1f}%)")
