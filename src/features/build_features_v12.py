"""
Feature set v12 — v7 + ARPA Lombardia ground-truth radiation for Sondrio.

ARPA sensor 2098 = Sondrio Fond. Fojanini (46.1676°N, 9.8509°E, 307m).
Located ~0.5 km from our household (46.17°N, 9.87°E).
Replaces Open-Meteo's 10-km-grid reanalysis radiation, which was bad in
the alpine valley microclimate.

Adds: arpa_radiation, arpa_radiation_lag1/4/96, arpa_roll96_mean,
      arpa_clearness (= arpa / clearsky_ghi), arpa_residual.
The original Open-Meteo shortwave_radiation feature is kept (v7 had it)
so the model can choose between them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
V7   = ROOT / "data/features/features_v7_all.parquet"
ARPA = ROOT / "data/external/arpa_sondrio_radiation.csv"
OUT  = ROOT / "data/features"
OUT.mkdir(parents=True, exist_ok=True)


def build():
    print("Loading v7 features...", flush=True)
    df = pd.read_parquet(V7)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  v7 rows={len(df)}  cols={df.shape[1]}", flush=True)

    print("Loading ARPA Sondrio radiation...", flush=True)
    arpa = pd.read_csv(ARPA, parse_dates=["timestamp"])
    print(f"  arpa rows={len(arpa)}  range {arpa['timestamp'].min()} -> {arpa['timestamp'].max()}", flush=True)

    # Merge — left join on timestamp
    df = df.merge(arpa, on="timestamp", how="left")
    n_missing = df["arpa_radiation"].isna().sum()
    print(f"  rows missing arpa_radiation after merge: {n_missing}/{len(df)}", flush=True)

    # Forward-fill / back-fill at edges
    df["arpa_radiation"] = df["arpa_radiation"].ffill().bfill()

    # Lag features
    df["arpa_lag1"]  = df["arpa_radiation"].shift(1)
    df["arpa_lag4"]  = df["arpa_radiation"].shift(4)
    df["arpa_lag96"] = df["arpa_radiation"].shift(96)

    # Rolling features (causal — use past only)
    df["arpa_roll4_mean"]  = df["arpa_radiation"].shift(1).rolling(window=4, min_periods=1).mean()
    df["arpa_roll96_mean"] = df["arpa_radiation"].shift(1).rolling(window=96, min_periods=1).mean()
    df["arpa_roll4_std"]   = df["arpa_radiation"].shift(1).rolling(window=4, min_periods=1).std().fillna(0)

    # Difference vs Open-Meteo grid radiation (where the alpine microclimate shows up)
    if "shortwave_radiation" in df.columns:
        df["arpa_minus_om"] = df["arpa_radiation"] - df["shortwave_radiation"]
        df["arpa_minus_om_lag1"] = df["arpa_minus_om"].shift(1)

    # Cleanness vs clear-sky physics (PHANN)
    if "clearsky_ghi" in df.columns:
        df["arpa_clearness"]  = df["arpa_radiation"] / (df["clearsky_ghi"].abs() + 1.0)
        df["arpa_residual"]   = df["arpa_radiation"] - df["clearsky_ghi"]

    print(f"  v12 rows={len(df)}  cols={df.shape[1]}", flush=True)
    return df


def get_feature_cols(df):
    drop = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
            "qow","hod","net_load","sell_price"}
    return [c for c in df.columns if c not in drop and not c.startswith("Unnamed")]


def get_train_val_test(df):
    train_mask = (df["timestamp"].dt.year == 2024) & (~df["timestamp"].dt.month.isin([4,9]))
    val_mask   = (df["timestamp"].dt.year == 2024) & (df["timestamp"].dt.month.isin([4,9]))
    test_mask  = (df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month.isin([4,9]))
    feats = get_feature_cols(df)
    return (df[train_mask].dropna(subset=feats).reset_index(drop=True),
            df[val_mask  ].dropna(subset=feats).reset_index(drop=True),
            df[test_mask ].dropna(subset=feats).reset_index(drop=True),
            feats)


if __name__ == "__main__":
    print("Building v12 features (v7 + ARPA local radiation)...", flush=True)
    df = build()
    feats = get_feature_cols(df)
    print(f"  Total cols: {df.shape[1]}  features: {len(feats)}", flush=True)

    df.to_parquet(OUT / "features_v12_all.parquet", index=False)
    train, val, test, _ = get_train_val_test(df)
    train.to_parquet(OUT / "features_v12_train.parquet", index=False)
    val  .to_parquet(OUT / "features_v12_val.parquet",   index=False)
    test .to_parquet(OUT / "features_v12_test.parquet",  index=False)
    (OUT / "features_v12_cols.txt").write_text("\n".join(feats))
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}", flush=True)

    v7_set = set((OUT / "features_v7_cols.txt").read_text().splitlines()) if (OUT / "features_v7_cols.txt").exists() else set()
    new_feats = [f for f in feats if f not in v7_set]
    print(f"\nNEW v12 features ({len(new_feats)}):")
    for f in new_feats: print(f"  + {f}")
