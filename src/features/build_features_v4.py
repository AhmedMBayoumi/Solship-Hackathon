"""
Feature set v4 — v2 + ONLY the useful v3 external features.
Drops noisy v3 features (importance < 100 in feature-importance audit).

KEEPS from v3:
  daylight_h, hrs_since_sunrise, hrs_until_sunset    (astronomy)
  dew_point_2m, apparent_temperature, app_hdd        (perceived temp)
  temp_d_lag1h                                        (1-hour temp swing)
  wind_speed_10m                                      (top wind feature)
  diffuse_radiation, direct_normal_irradiance         (radiation breakdown)
  precip_24h_sum                                      (24h precipitation)

DROPS from v3:
  All calendar/social flags (low importance)
  Snowfall variants (Sondrio Apr+Sep has no snow)
  is_dinner_hour, is_school_holiday, is_bridge_day, etc.
  Most interaction terms
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROCESSED = ROOT / "data/processed/dataset_processed.csv"
WEATHER_EXT = ROOT / "data/external/sondrio_weather_extended.csv"
OUT_DIR = ROOT / "data/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_HOLIDAYS = {(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)}
EASTER_MONDAYS = {2024:(4,1), 2025:(4,21)}
LOCAL_HOLIDAYS = {(6,19)}


def is_holiday(ts):
    md = (ts.month, ts.day)
    em = EASTER_MONDAYS.get(ts.year)
    return md in FIXED_HOLIDAYS or md in LOCAL_HOLIDAYS or (em and md == em)


def add_fourier(df):
    df = df.copy()
    t_min = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    for h, lab in [(24, "24h"), (12, "12h"), (8, "8h"), (6, "6h"), (4, "4h")]:
        p = h * 60
        df[f"sin_{lab}"] = np.sin(2 * np.pi * t_min / p)
        df[f"cos_{lab}"] = np.cos(2 * np.pi * t_min / p)
    doy = df["timestamp"].dt.day_of_year
    df["sin_annual"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_annual"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_semiann"] = np.sin(4 * np.pi * doy / 365.25)
    df["cos_semiann"] = np.cos(4 * np.pi * doy / 365.25)
    return df


# Curated list of useful v3 features (importance >= 50 from audit)
USEFUL_NEW_FEATURES = [
    "daylight_h",
    "dew_point_2m",
    "hrs_since_sunrise",
    "apparent_temperature",
    "hrs_until_sunset",
    "temp_d_lag1h",
    "app_hdd",
    "wind_speed_10m",
    "wind_lag1",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "pressure_msl",
    "surface_pressure",
    "precip_24h_sum",
    "wind_gusts_10m",
    "direct_radiation",
    "app_cdd",
]


def build():
    df = pd.read_csv(PROCESSED, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Calendar (v2 base)
    df["hour"]  = df["timestamp"].dt.hour
    df["minute"]= df["timestamp"].dt.minute
    df["dow"]   = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.day_of_year
    df["week_of_year"]= df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["is_holiday"]  = df["timestamp"].apply(is_holiday).astype(int)
    df["tariff_enc"]  = df["tariff_band"].map({"F1":0,"F2":1,"F3":2}).fillna(0).astype(int)

    df["hour_x_wknd"] = df["hour"] * df["is_weekend"]
    df["hour_x_hol"]  = df["hour"] * df["is_holiday"]

    df = add_fourier(df)

    # Lags (load) — same as v2/v3
    LAGS = [1,2,3,4,6,8,12,16,24,32,48,64,96,192,288,384,480,576,672,1344,2016]
    for lag in LAGS:
        df[f"lag_{lag}"] = df["load_kw"].shift(lag)
    df["d_lag1"]   = df["lag_1"]   - df["lag_2"]
    df["d_lag4"]   = df["lag_4"]   - df["lag_8"]
    df["d_lag96"]  = df["lag_96"]  - df["lag_192"]
    df["d_lag672"] = df["lag_672"] - df["lag_1344"]

    for lag in [1, 4, 8, 96, 192, 672]:
        df[f"pv_lag{lag}"] = df["pv_kw"].shift(lag)

    for w in [4, 8, 16, 96, 384, 672]:
        df[f"roll_{w}_mean"] = df["load_kw"].shift(1).rolling(w).mean()
    for w in [4, 16, 96]:
        df[f"roll_{w}_std"]  = df["load_kw"].shift(1).rolling(w).std()
        df[f"roll_{w}_max"]  = df["load_kw"].shift(1).rolling(w).max()
        df[f"roll_{w}_min"]  = df["load_kw"].shift(1).rolling(w).min()

    df["net_load_lag1"]   = df["net_load"].shift(1)
    df["net_load_lag4"]   = df["net_load"].shift(4)
    df["net_load_lag96"]  = df["net_load"].shift(96)
    df["net_load_lag672"] = df["net_load"].shift(672)
    df["net_load_roll96_mean"] = df["net_load"].shift(1).rolling(96).mean()

    # Extended weather merge — keep only useful columns + base v2 weather
    if WEATHER_EXT.exists():
        wx = pd.read_csv(WEATHER_EXT, parse_dates=["timestamp"])
        wx["timestamp"] = wx["timestamp"].dt.round("15min")
        wx = wx.groupby("timestamp").first().reset_index()
        # Always include base v2 weather + the curated useful v3 features
        base_w = ["temperature_2m","shortwave_radiation","cloud_cover","relative_humidity_2m","hdd","cdd"]
        cols = ["timestamp"] + base_w + USEFUL_NEW_FEATURES
        cols = [c for c in cols if c in wx.columns or c == "timestamp"]
        df = df.merge(wx[cols], on="timestamp", how="left")
        # Forward/back fill any NaN
        for c in cols[1:]:
            if c in df.columns and df[c].isna().any():
                df[c] = df[c].fillna(method="ffill").fillna(method="bfill").fillna(0)
        # Weather lags
        df["temp_lag1"]     = df["temperature_2m"].shift(1)
        df["temp_lag96"]    = df["temperature_2m"].shift(96)
        df["temp_lag672"]   = df["temperature_2m"].shift(672)
        df["temp_d_lag96"]  = df["temperature_2m"] - df["temp_lag96"]
        df["rad_lag96"]     = df["shortwave_radiation"].shift(96)
        df["rad_roll96_mean"] = df["shortwave_radiation"].rolling(96).mean()
        # ONE useful interaction (cold dinner hour)
        h_dec = df["hour"] + df["minute"] / 60.0
        df["cold_evening"] = ((df["temperature_2m"] < 10) & (h_dec.between(19.5, 21.5))).astype(int)
        df["temp_x_wknd"]  = df["temperature_2m"] * df["is_weekend"]
        df["hdd_x_wknd"]   = df["hdd"] * df["is_weekend"]

    return df


def get_feature_cols(df):
    drop = {"timestamp", "load_kw", "p_battery_kw", "grid_kw", "tariff_band", "minute",
            "qow", "hod", "net_load", "sell_price"}
    return [c for c in df.columns if c not in drop and not c.startswith("Unnamed")]


def get_train_val_test(df):
    train_mask = (df["timestamp"].dt.year == 2024) & (~df["timestamp"].dt.month.isin([4, 9]))
    val_mask   = (df["timestamp"].dt.year == 2024) & (df["timestamp"].dt.month.isin([4, 9]))
    test_mask  = (df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month.isin([4, 9]))
    feats = get_feature_cols(df)
    return (df[train_mask].dropna(subset=feats).reset_index(drop=True),
            df[val_mask  ].dropna(subset=feats).reset_index(drop=True),
            df[test_mask ].dropna(subset=feats).reset_index(drop=True),
            feats)


if __name__ == "__main__":
    print("Building feature set v4 (v2 + only useful v3 externals)...")
    df = build()
    feats = get_feature_cols(df)
    print(f"  Total rows: {len(df)}  cols: {len(df.columns)}")
    print(f"  Features (v4): {len(feats)}")

    df.to_parquet(OUT_DIR / "features_v4_all.parquet", index=False)
    train, val, test, _ = get_train_val_test(df)
    train.to_parquet(OUT_DIR / "features_v4_train.parquet", index=False)
    val.to_parquet  (OUT_DIR / "features_v4_val.parquet",   index=False)
    test.to_parquet (OUT_DIR / "features_v4_test.parquet",  index=False)
    (OUT_DIR / "features_v4_cols.txt").write_text("\n".join(feats))
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  Saved -> {OUT_DIR}/features_v4_*.parquet")
