"""
Feature set v6 — v5 + user-suggested "High-Impact" Italian behaviour features
+ feature-selection-friendly redundancy reduction.

NEW features (vs v5):
  CULTURAL / ITALIAN PULSE:
    - it_morning_rush      (07:30-09:30 binary)
    - it_lunch_peak        (12:30-14:00 binary, replaces is_lunch_hour)
    - it_tou_shift         (21:00-23:00 binary, F3 delayed-appliance start)

  THERMAL THRESHOLDS:
    - cooling_load_244     (max(0, temp - 24.4) — Italian residential trigger)
    - alpine_temp_delta    (T(t) - T(t-4) — Föhn detector, 1h gradient)
    - htí_apparent         (humidity-temp index using apparent_temp)

  SIGNAL DECOMPOSITION:
    - baseload_24h         (24h rolling min of load — fridge/standby proxy)
    - load_range_4h        (max - min over 4h — volatility envelope)
    - load_entropy_8       (Shannon entropy of last 8 load bins — chaos proxy)

  CROSS-DOMAIN:
    - is_empty_proxy       (high PV + low load = empty house)
    - solar_lag_3h         (GHI shifted 3h — passive thermal gain)
    - net_load_baseload    (net load - 24h baseload — true demand vs base)

  RECURSIVE STATE:
    - is_climbing          (lag_1 > lag_2)
    - is_high_state        (lag_1 > 2 kW = high-load regime)
    - lagged_var_1h        (std of last 4 actuals)

We KEEP the v5 base since v5 features dominate importance (90%+ from lags).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROCESSED   = ROOT / "data/processed/dataset_processed.csv"
WEATHER_EXT = ROOT / "data/external/sondrio_weather_extended.csv"
CLEARSKY    = ROOT / "data/external/sondrio_clearsky.csv"
OUT_DIR     = ROOT / "data/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_HOLIDAYS = {(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)}
EASTER_MONDAYS = {2024:(4,1), 2025:(4,21)}
LOCAL_HOLIDAYS = {(6,19)}


def is_holiday(ts):
    md = (ts.month, ts.day)
    em = EASTER_MONDAYS.get(ts.year)
    return md in FIXED_HOLIDAYS or md in LOCAL_HOLIDAYS or (em and md == em)


def is_bridge_day(ts):
    if ts.dayofweek >= 5 or is_holiday(ts):
        return False
    next_day, prev_day = ts + pd.Timedelta(days=1), ts - pd.Timedelta(days=1)
    next_off = is_holiday(next_day) or next_day.dayofweek >= 5
    prev_off = is_holiday(prev_day) or prev_day.dayofweek >= 5
    if ts.dayofweek == 0 and next_off: return True
    if ts.dayofweek == 4 and prev_off: return True
    return False


def add_fourier(df):
    df = df.copy()
    t_min = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    for h, lab in [(24, "24h"), (12, "12h"), (8, "8h"), (6, "6h"), (4, "4h")]:
        df[f"sin_{lab}"] = np.sin(2 * np.pi * t_min / (h*60))
        df[f"cos_{lab}"] = np.cos(2 * np.pi * t_min / (h*60))
    doy = df["timestamp"].dt.day_of_year
    df["sin_annual"]  = np.sin(2 * np.pi * doy / 365.25)
    df["cos_annual"]  = np.cos(2 * np.pi * doy / 365.25)
    df["sin_semiann"] = np.sin(4 * np.pi * doy / 365.25)
    df["cos_semiann"] = np.cos(4 * np.pi * doy / 365.25)
    return df


# Useful curated v3 weather features kept from v5
USEFUL_WEATHER = [
    "daylight_h", "dew_point_2m", "hrs_since_sunrise", "apparent_temperature",
    "hrs_until_sunset", "temp_d_lag1h", "app_hdd",
    "wind_speed_10m", "wind_lag1", "direct_normal_irradiance", "diffuse_radiation",
    "pressure_msl", "surface_pressure", "precip_24h_sum",
    "wind_gusts_10m", "direct_radiation", "app_cdd",
]

# PHANN columns from v5
CLEARSKY_COLS = ["solar_zenith", "solar_elevation", "solar_azimuth",
                 "clearsky_ghi", "clearsky_dni", "clearsky_dhi",
                 "cos_solar_zenith", "is_daytime_phys"]


def build():
    df = pd.read_csv(PROCESSED, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Calendar
    df["hour"]   = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dow"]    = df["timestamp"].dt.dayofweek
    df["month"]  = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.day_of_year
    df["week_of_year"]= df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["is_holiday"]  = df["timestamp"].apply(is_holiday).astype(int)
    df["is_bridge_day"] = df["timestamp"].apply(is_bridge_day).astype(int)
    df["tariff_enc"]  = df["tariff_band"].map({"F1":0,"F2":1,"F3":2}).fillna(0).astype(int)

    df["hour_x_wknd"] = df["hour"] * df["is_weekend"]
    df["hour_x_hol"]  = df["hour"] * df["is_holiday"]
    df = add_fourier(df)

    # ── NEW: Italian Pulse (cultural/behavioural) ───────────────
    h_dec = df["hour"] + df["minute"]/60.0
    df["it_morning_rush"] = ((h_dec >= 7.5)  & (h_dec <= 9.5 )).astype(int)
    df["it_lunch_peak"]   = ((h_dec >= 12.5) & (h_dec <= 14.0)).astype(int)
    df["it_tou_shift"]    = ((h_dec >= 21.0) & (h_dec <= 23.0)).astype(int)
    df["it_dinner_hour"]  = ((h_dec >= 19.5) & (h_dec <= 21.5)).astype(int)  # keep from v3 too
    df["it_pre_dawn"]     = ((h_dec >= 5.0)  & (h_dec <= 7.0 )).astype(int)

    # ── Lags / rolling — same as v5 ─────────────────────────────
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

    # ── NEW: Signal decomposition ───────────────────────────────
    df["baseload_24h"]   = df["load_kw"].shift(1).rolling(96).min()       # 24h rolling min
    df["baseload_1week"] = df["load_kw"].shift(1).rolling(672).min()      # 1-week rolling min
    df["load_range_4h"]  = (df["load_kw"].shift(1).rolling(16).max()
                            - df["load_kw"].shift(1).rolling(16).min())   # 4h volatility envelope
    df["load_range_24h"] = (df["load_kw"].shift(1).rolling(96).max()
                            - df["load_kw"].shift(1).rolling(96).min())

    # Approximate Shannon entropy in last 8 timesteps (using histogram of bins)
    def rolling_entropy(arr, window=8, bins=4):
        out = np.full(len(arr), np.nan)
        for i in range(window, len(arr)):
            window_vals = arr[i-window:i]
            if np.all(np.isnan(window_vals)):
                continue
            hist, _ = np.histogram(window_vals, bins=bins, range=(0, 6))
            p = hist / hist.sum() if hist.sum() > 0 else np.zeros(bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.nansum(p * np.log(p + 1e-9))
            out[i] = ent
        return out
    df["load_entropy_8"]  = rolling_entropy(df["load_kw"].shift(1).values, window=8,  bins=4)
    df["load_entropy_16"] = rolling_entropy(df["load_kw"].shift(1).values, window=16, bins=5)

    # ── NEW: Recursive state ────────────────────────────────────
    df["is_climbing"]   = (df["lag_1"] > df["lag_2"]).astype(int)
    df["is_high_state"] = (df["lag_1"] > 2.0).astype(int)
    df["lagged_var_1h"] = df["load_kw"].shift(1).rolling(4).std()
    df["lagged_var_2h"] = df["load_kw"].shift(1).rolling(8).std()

    # ── Weather merge ───────────────────────────────────────────
    if WEATHER_EXT.exists():
        wx = pd.read_csv(WEATHER_EXT, parse_dates=["timestamp"])
        wx["timestamp"] = wx["timestamp"].dt.round("15min")
        wx = wx.groupby("timestamp").first().reset_index()
        base_w = ["temperature_2m","shortwave_radiation","cloud_cover","relative_humidity_2m","hdd","cdd"]
        cols = ["timestamp"] + base_w + USEFUL_WEATHER
        cols = [c for c in cols if c in wx.columns or c == "timestamp"]
        df = df.merge(wx[cols], on="timestamp", how="left")
        for c in cols[1:]:
            if c in df.columns and df[c].isna().any():
                df[c] = df[c].fillna(method="ffill").fillna(method="bfill").fillna(0)
        df["temp_lag1"]    = df["temperature_2m"].shift(1)
        df["temp_lag96"]   = df["temperature_2m"].shift(96)
        df["temp_d_lag96"] = df["temperature_2m"] - df["temp_lag96"]
        df["rad_lag96"]    = df["shortwave_radiation"].shift(96)
        df["rad_roll96_mean"] = df["shortwave_radiation"].rolling(96).mean()

        # ── NEW: Italian thermal thresholds ─────────────────────
        df["cooling_load_244"] = (df["temperature_2m"] - 24.4).clip(lower=0)
        df["heating_load_18"]  = (18.0 - df["temperature_2m"]).clip(lower=0)
        # Föhn: 1-hour temperature gradient (T(t) - T(t-4))
        df["alpine_temp_delta_1h"] = df["temperature_2m"] - df["temperature_2m"].shift(4)
        df["alpine_temp_delta_3h"] = df["temperature_2m"] - df["temperature_2m"].shift(12)
        # Humidity-Temperature Index (Italian Lombardy)
        # HTI = T + 0.5 * RH/100 * T  (approx)
        df["hti"] = df["temperature_2m"] + 0.5 * (df["relative_humidity_2m"] / 100.0) * df["temperature_2m"]
        # Apparent temp HDD/CDD (already in v5 feature names)
        if "apparent_temperature" in df.columns:
            df["app_hdd"] = (18 - df["apparent_temperature"]).clip(lower=0)
            df["app_cdd"] = (df["apparent_temperature"] - 24).clip(lower=0)

    # ── PHANN clear-sky merge ───────────────────────────────────
    if CLEARSKY.exists():
        cs = pd.read_csv(CLEARSKY, parse_dates=["timestamp"])
        cs["timestamp"] = cs["timestamp"].dt.round("15min")
        cs = cs.groupby("timestamp").first().reset_index()
        df = df.merge(cs[["timestamp"] + CLEARSKY_COLS], on="timestamp", how="left")
        for c in CLEARSKY_COLS:
            if df[c].isna().any():
                df[c] = df[c].fillna(method="ffill").fillna(method="bfill").fillna(0)
        eps = 5.0
        if "shortwave_radiation" in df.columns:
            df["rad_clearness"] = df["shortwave_radiation"] / (df["clearsky_ghi"] + eps)
            df["rad_residual"]  = df["shortwave_radiation"] - df["clearsky_ghi"]
        if "direct_normal_irradiance" in df.columns:
            df["dni_clearness"] = df["direct_normal_irradiance"] / (df["clearsky_dni"] + eps)
        df["pv_per_clearsky"] = df["pv_kw"] / (df["clearsky_ghi"] + eps)
        df["pv_per_clearsky_lag1"] = df["pv_per_clearsky"].shift(1)

        # ── NEW: solar lag 3h (passive thermal gain) ────────────
        df["solar_lag_3h"]  = df["shortwave_radiation"].shift(12)
        df["solar_lag_6h"]  = df["shortwave_radiation"].shift(24)
        df["clearsky_lag_3h"] = df["clearsky_ghi"].shift(12)

    # ── NEW: Cross-domain — "Ghost House" / occupancy proxy ─────
    # Compare current PV to recent max; compare load to baseload.
    df["pv_vs_max"]      = df["pv_kw"] / (df["pv_kw"].shift(1).rolling(96).max() + 0.5)
    df["load_vs_base"]   = df["load_kw"].shift(1) / (df["baseload_24h"] + 0.1)
    df["is_empty_proxy"] = ((df["pv_vs_max"] > 0.5) & (df["load_vs_base"] < 1.2)).astype(int)
    df["net_load_baseload"] = df["lag_1"] - df["baseload_24h"]   # demand above base

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
    print("Building feature set v6 (Italian behaviour + signal decomposition + occupancy)...")
    df = build()
    feats = get_feature_cols(df)
    print(f"  Total rows: {len(df)}  cols: {len(df.columns)}  features: {len(feats)}")
    df.to_parquet(OUT_DIR / "features_v6_all.parquet", index=False)
    train, val, test, _ = get_train_val_test(df)
    train.to_parquet(OUT_DIR / "features_v6_train.parquet", index=False)
    val.to_parquet  (OUT_DIR / "features_v6_val.parquet",   index=False)
    test.to_parquet (OUT_DIR / "features_v6_test.parquet",  index=False)
    (OUT_DIR / "features_v6_cols.txt").write_text("\n".join(feats))
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  Saved -> {OUT_DIR}/features_v6_*.parquet")

    v5 = set((OUT_DIR / "features_v5_cols.txt").read_text().splitlines()) if (OUT_DIR / "features_v5_cols.txt").exists() else set()
    new_feats = [f for f in feats if f not in v5]
    print(f"\nNEW features in v6 ({len(new_feats)}):")
    for f in new_feats: print(f"  + {f}")
