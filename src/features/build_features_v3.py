"""
Feature set v3 — adds external signals targeted at SPIKE detection.

New beyond v2:
  EXTENDED WEATHER (free, Open-Meteo):
    wind_speed/gusts/direction, precipitation/rain/snowfall,
    surface_pressure, dew_point, apparent_temperature,
    direct/diffuse/DNI radiation, is_day flag,
    daylight_h, hrs_since_sunrise, hrs_until_sunset.

  CALENDAR/SOCIAL (free, computed):
    italian school-holiday flag, days_since/until_holiday,
    bridge-day flag (ponte), heating-season flag,
    italian dinner-hour flag, late-evening tail flag,
    pre-dawn flag, christmas/easter/ferragosto window flags,
    DST transition flag.

  WEATHER LAGS (more):
    temperature 24h-difference, snowfall_lag1, wind_speed_lag1,
    rolling 24h precipitation total, rolling 24h snowfall total.

  WEATHER × CALENDAR INTERACTIONS:
    cold_evening (T<10 AND hour 18-22), HDD × is_weekend,
    snowfall × is_holiday, wind × hour.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROCESSED = ROOT / "data/processed/dataset_processed.csv"
WEATHER_EXT = ROOT / "data/external/sondrio_weather_extended.csv"
OUT_DIR = ROOT / "data/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Italian holidays (more comprehensive than v2) ──
FIXED_HOLIDAYS = {(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)}
EASTER_MONDAYS = {2024:(4,1), 2025:(4,21)}
LOCAL_HOLIDAYS = {(6,19)}  # Sondrio patron saint

# Italian school holiday windows (Lombardy region)
SCHOOL_HOLIDAY_WINDOWS = [
    (2024, 1, 1, 2024, 1, 7),     # Christmas break end
    (2024, 3, 28, 2024, 4, 2),    # Easter break 2024
    (2024, 4, 25, 2024, 4, 28),   # Liberation extended
    (2024, 6, 8, 2024, 9, 12),    # Summer holiday
    (2024, 11, 1, 2024, 11, 3),   # All Saints
    (2024, 12, 23, 2024, 12, 31), # Christmas start
    (2025, 1, 1, 2025, 1, 7),
    (2025, 4, 17, 2025, 4, 23),   # Easter 2025
    (2025, 4, 25, 2025, 4, 28),
    (2025, 6, 7, 2025, 9, 11),    # Summer 2025
    (2025, 11, 1, 2025, 11, 3),
    (2025, 12, 23, 2025, 12, 31),
]


def is_holiday(ts):
    md = (ts.month, ts.day)
    em = EASTER_MONDAYS.get(ts.year)
    return md in FIXED_HOLIDAYS or md in LOCAL_HOLIDAYS or (em and md == em)


def in_school_holiday(ts):
    for sy, sm, sd, ey, em, ed in SCHOOL_HOLIDAY_WINDOWS:
        if pd.Timestamp(sy, sm, sd) <= ts <= pd.Timestamp(ey, em, ed):
            return True
    return False


def days_to_next_holiday(ts):
    """Days from `ts` to the next Italian holiday (max 30)."""
    for d in range(31):
        t = ts + pd.Timedelta(days=d)
        if is_holiday(t) or t.dayofweek >= 5:
            return d
    return 30


def days_since_last_holiday(ts):
    for d in range(31):
        t = ts - pd.Timedelta(days=d)
        if is_holiday(t) or t.dayofweek >= 5:
            return d
    return 30


def is_bridge_day(ts):
    """Italian 'ponte': workday between holiday and weekend."""
    if ts.dayofweek >= 5 or is_holiday(ts):
        return False
    next_day = ts + pd.Timedelta(days=1)
    prev_day = ts - pd.Timedelta(days=1)
    next_is_off = is_holiday(next_day) or next_day.dayofweek >= 5
    prev_is_off = is_holiday(prev_day) or prev_day.dayofweek >= 5
    if ts.dayofweek == 0 and next_is_off:    # Mon, with Tue off
        return True
    if ts.dayofweek == 4 and prev_is_off:    # Fri, with Thu off
        return True
    return False


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


def build():
    df = pd.read_csv(PROCESSED, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ── Calendar base ─────────────────────────────────────────
    df["hour"]  = df["timestamp"].dt.hour
    df["minute"]= df["timestamp"].dt.minute
    df["dow"]   = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.day_of_year
    df["week_of_year"]= df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["is_holiday"]  = df["timestamp"].apply(is_holiday).astype(int)
    df["tariff_enc"]  = df["tariff_band"].map({"F1":0,"F2":1,"F3":2}).fillna(0).astype(int)

    # NEW calendar/social features
    print("Adding calendar/social features...")
    df["is_school_holiday"] = df["timestamp"].apply(in_school_holiday).astype(int)
    df["is_bridge_day"]     = df["timestamp"].apply(is_bridge_day).astype(int)
    df["days_to_next_hol"]  = df["timestamp"].apply(days_to_next_holiday)
    df["days_since_last_hol"] = df["timestamp"].apply(days_since_last_holiday)

    # Italian dinner hour: 19:30-21:30 typical mealtime spike
    h_dec = df["hour"] + df["minute"] / 60.0
    df["is_dinner_hour"]   = ((h_dec >= 19.5) & (h_dec <= 21.5)).astype(int)
    df["is_lunch_hour"]    = ((h_dec >= 12.0) & (h_dec <= 13.5)).astype(int)
    df["is_late_evening"]  = ((h_dec >= 21.5) & (h_dec <= 23.5)).astype(int)
    df["is_pre_dawn"]      = ((h_dec >= 5.0) & (h_dec <= 7.0)).astype(int)

    # Heating season Italian convention (Oct 15 → Apr 15)
    df["is_heating_season"] = (
        ((df["month"] >= 10) & ((df["month"] > 10) | (df["timestamp"].dt.day >= 15))) |
        ((df["month"] <= 4)  & ((df["month"] <  4) | (df["timestamp"].dt.day <= 15)))
    ).astype(int)
    df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)

    # Special multi-day windows
    df["is_christmas_window"] = (
        ((df["month"] == 12) & (df["timestamp"].dt.day >= 23)) |
        ((df["month"] == 1)  & (df["timestamp"].dt.day <= 6))
    ).astype(int)
    df["is_ferragosto_window"] = (
        ((df["month"] == 8) & (df["timestamp"].dt.day >= 10) & (df["timestamp"].dt.day <= 20))
    ).astype(int)

    # Fourier
    df = add_fourier(df)

    # ── Lags (load) — same as v2 ──────────────────────────────
    LAGS = [1,2,3,4,6,8,12,16,24,32,48,64,96,192,288,384,480,576,672,1344,2016]
    for lag in LAGS:
        df[f"lag_{lag}"] = df["load_kw"].shift(lag)
    df["d_lag1"]   = df["lag_1"]   - df["lag_2"]
    df["d_lag4"]   = df["lag_4"]   - df["lag_8"]
    df["d_lag96"]  = df["lag_96"]  - df["lag_192"]
    df["d_lag672"] = df["lag_672"] - df["lag_1344"]

    # PV lags
    for lag in [1, 4, 8, 96, 192, 672]:
        df[f"pv_lag{lag}"] = df["pv_kw"].shift(lag)

    # Rolling stats
    for w in [4, 8, 16, 96, 384, 672]:
        df[f"roll_{w}_mean"] = df["load_kw"].shift(1).rolling(w).mean()
    for w in [4, 16, 96]:
        df[f"roll_{w}_std"]  = df["load_kw"].shift(1).rolling(w).std()
        df[f"roll_{w}_max"]  = df["load_kw"].shift(1).rolling(w).max()
        df[f"roll_{w}_min"]  = df["load_kw"].shift(1).rolling(w).min()

    # Net load
    df["net_load_lag1"]   = df["net_load"].shift(1)
    df["net_load_lag4"]   = df["net_load"].shift(4)
    df["net_load_lag96"]  = df["net_load"].shift(96)
    df["net_load_lag672"] = df["net_load"].shift(672)
    df["net_load_roll96_mean"] = df["net_load"].shift(1).rolling(96).mean()

    # ── Extended weather merge ────────────────────────────────
    if WEATHER_EXT.exists():
        print("Merging extended weather...")
        wx = pd.read_csv(WEATHER_EXT, parse_dates=["timestamp"])
        wx["timestamp"] = wx["timestamp"].dt.round("15min")
        wx = wx.groupby("timestamp").first().reset_index()
        df = df.merge(wx, on="timestamp", how="left")
        # Fill NaN with reasonable defaults
        for c in wx.columns:
            if c == "timestamp": continue
            if df[c].isna().any():
                df[c] = df[c].fillna(method="ffill").fillna(method="bfill").fillna(0)
        # Weather lags
        df["temp_lag1"]    = df["temperature_2m"].shift(1)
        df["temp_lag96"]   = df["temperature_2m"].shift(96)
        df["temp_lag672"]  = df["temperature_2m"].shift(672)
        df["temp_d_lag96"] = df["temperature_2m"] - df["temp_lag96"]
        df["temp_d_lag1h"] = df["temperature_2m"] - df["temperature_2m"].shift(4)  # 1-hour temp change
        df["rad_lag96"]    = df["shortwave_radiation"].shift(96)
        df["rad_roll96_mean"] = df["shortwave_radiation"].rolling(96).mean()
        df["wind_lag1"]    = df["wind_speed_10m"].shift(1)
        df["snowfall_lag1"]= df["snowfall"].shift(1)
        df["precip_24h_sum"]   = df["precipitation"].rolling(96).sum()
        df["snowfall_24h_sum"] = df["snowfall"].rolling(96).sum()
        # Apparent temp HDD (perceived cold drives HVAC)
        df["app_hdd"] = (18 - df["apparent_temperature"]).clip(lower=0)
        df["app_cdd"] = (df["apparent_temperature"] - 24).clip(lower=0)

        # Interactions targeted at spike detection
        df["cold_evening"]      = ((df["temperature_2m"] < 10) & df["is_dinner_hour"]).astype(int)
        df["very_cold_morning"] = ((df["temperature_2m"] < 5) & df["is_pre_dawn"]).astype(int)
        df["hot_afternoon"]     = ((df["temperature_2m"] > 25) & (df["hour"].between(13,18))).astype(int)
        df["hdd_x_wknd"]        = df["hdd"] * df["is_weekend"]
        df["cdd_x_summer"]      = df["cdd"] * df["is_summer"]
        df["snowfall_x_holiday"]= df["snowfall"] * df["is_holiday"]
        df["wind_x_hour"]       = df["wind_speed_10m"] * df["hour"]
        df["temp_x_wknd"]       = df["temperature_2m"] * df["is_weekend"]
        df["rad_x_dinner"]      = df["shortwave_radiation"] * df["is_dinner_hour"]
    else:
        print(f"  WARNING: extended weather file not found at {WEATHER_EXT}")
        for c in ["temperature_2m","shortwave_radiation","cloud_cover","relative_humidity_2m",
                  "wind_speed_10m","precipitation","snowfall","surface_pressure",
                  "dew_point_2m","apparent_temperature","is_day",
                  "hrs_since_sunrise","hrs_until_sunset","daylight_h","hdd","cdd"]:
            df[c] = 0.0

    return df


def get_feature_cols(df):
    """v3 feature columns. Excludes targets, leaky data, and identifiers."""
    drop = {"timestamp", "load_kw", "p_battery_kw", "grid_kw", "tariff_band", "minute",
            "qow", "hod", "net_load", "sell_price"}
    return [c for c in df.columns if c not in drop and not c.startswith("Unnamed")]


def get_train_val_test(df):
    """Default split: 2024 except Apr+Sep for train, 2024 Apr+Sep for val, 2025 Apr+Sep for test."""
    train_mask = (df["timestamp"].dt.year == 2024) & (~df["timestamp"].dt.month.isin([4, 9]))
    val_mask   = (df["timestamp"].dt.year == 2024) & (df["timestamp"].dt.month.isin([4, 9]))
    test_mask  = (df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month.isin([4, 9]))
    feats = get_feature_cols(df)
    train = df[train_mask].dropna(subset=feats).reset_index(drop=True)
    val   = df[val_mask  ].dropna(subset=feats).reset_index(drop=True)
    test  = df[test_mask ].dropna(subset=feats).reset_index(drop=True)
    return train, val, test, feats


if __name__ == "__main__":
    print("Building enhanced feature set v3 with external signals...")
    df = build()
    feats = get_feature_cols(df)
    print(f"  Total rows: {len(df)}  cols: {len(df.columns)}")
    print(f"  Features (v3): {len(feats)}")

    df.to_parquet(OUT_DIR / "features_v3_all.parquet", index=False)
    train, val, test, _ = get_train_val_test(df)
    train.to_parquet(OUT_DIR / "features_v3_train.parquet", index=False)
    val.to_parquet  (OUT_DIR / "features_v3_val.parquet",   index=False)
    test.to_parquet (OUT_DIR / "features_v3_test.parquet",  index=False)
    (OUT_DIR / "features_v3_cols.txt").write_text("\n".join(feats))
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  Saved -> {OUT_DIR}/features_v3_*.parquet")

    # Print new features
    v2_feats = set((OUT_DIR / "features_v2_cols.txt").read_text().splitlines()) if (OUT_DIR / "features_v2_cols.txt").exists() else set()
    new_feats = [f for f in feats if f not in v2_feats]
    print(f"\nNEW features in v3 ({len(new_feats)}):")
    for f in new_feats: print(f"  + {f}")
