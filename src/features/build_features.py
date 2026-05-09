"""
Build the full feature matrix for training and inference.
Reads processed dataset + weather, outputs parquet files per year.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).parents[2] / "data" / "processed" / "dataset_processed.csv"
WEATHER    = Path(__file__).parents[2] / "data" / "external"  / "sondrio_weather.csv"
OUT_DIR    = Path(__file__).parents[2] / "data" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY = 16.0
FIXED_HOLIDAYS = {(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)}
EASTER_MONDAYS = {2024:(4,1), 2025:(4,21)}
# Sondrio patron saint
LOCAL_HOLIDAYS = {(6,19)}


def is_holiday(ts: pd.Series) -> pd.Series:
    results = []
    for t in ts:
        md = (t.month, t.day)
        em = EASTER_MONDAYS.get(t.year, None)
        flag = md in FIXED_HOLIDAYS or md in LOCAL_HOLIDAYS or (em and md == em)
        results.append(int(flag))
    return pd.Series(results, index=ts.index)


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mins_in_day = 24 * 60
    t_min = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    for period_h, label in [(24, "24h"), (12, "12h"), (8, "8h")]:
        period_min = period_h * 60
        df[f"sin_{label}"] = np.sin(2 * np.pi * t_min / period_min)
        df[f"cos_{label}"] = np.cos(2 * np.pi * t_min / period_min)
    # Annual seasonality
    doy = df["timestamp"].dt.day_of_year
    days_year = 365.25
    df["sin_annual"] = np.sin(2 * np.pi * doy / days_year)
    df["cos_annual"] = np.cos(2 * np.pi * doy / days_year)
    return df


def build(use_weather: bool = True) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Calendar features ────────────────────────────────────────────────
    df["hour"]        = df["timestamp"].dt.hour
    df["minute"]      = df["timestamp"].dt.minute
    df["dow"]         = df["timestamp"].dt.dayofweek       # 0=Mon
    df["month"]       = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.day_of_year
    df["week_of_year"]= df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["is_holiday"]  = is_holiday(df["timestamp"])
    df["tariff_enc"]  = df["tariff_band"].map({"F1": 0, "F2": 1, "F3": 2}).fillna(0).astype(int)

    # ── Fourier encodings ────────────────────────────────────────────────
    df = add_fourier(df)

    # ── Lag features (load) ───────────────────────────────────────────────
    for lag in [1, 4, 8, 96, 192, 672]:
        df[f"lag_{lag}"] = df["load_kw"].shift(lag)

    # ── Lag features (PV — useful for net load context) ──────────────────
    df["pv_lag1"]  = df["pv_kw"].shift(1)
    df["pv_lag96"] = df["pv_kw"].shift(96)

    # ── Rolling means (load) ─────────────────────────────────────────────
    for window in [4, 16, 96]:
        df[f"roll_{window}_mean"] = df["load_kw"].shift(1).rolling(window).mean()

    # Rolling std for recent volatility
    df["roll_4_std"]  = df["load_kw"].shift(1).rolling(4).std()
    df["roll_96_std"] = df["load_kw"].shift(1).rolling(96).std()

    # ── Net load lag ──────────────────────────────────────────────────────
    df["net_load_lag1"]  = df["net_load"].shift(1)
    df["net_load_lag96"] = df["net_load"].shift(96)

    # ── Weather ───────────────────────────────────────────────────────────
    if use_weather and WEATHER.exists():
        wx = pd.read_csv(WEATHER, parse_dates=["timestamp"])
        # Round weather timestamps to nearest 15min to align
        wx["timestamp"] = wx["timestamp"].dt.round("15min")
        wx = wx.groupby("timestamp").first().reset_index()
        df = df.merge(wx[["timestamp","temperature_2m","shortwave_radiation",
                           "cloud_cover","relative_humidity_2m","hdd","cdd"]],
                      on="timestamp", how="left")
        # Add weather lags
        df["temp_lag96"]  = df["temperature_2m"].shift(96)
        df["rad_lag96"]   = df["shortwave_radiation"].shift(96)
        print("  Weather columns merged OK")
    else:
        # Fill with zeros so pipeline works without weather
        for col in ["temperature_2m","shortwave_radiation","cloud_cover",
                    "relative_humidity_2m","hdd","cdd","temp_lag96","rad_lag96"]:
            df[col] = 0.0
        if use_weather:
            print("  WARNING: weather file not found, using zeros")

    return df


FEATURE_COLS = [
    # Lags
    "lag_1","lag_4","lag_8","lag_96","lag_192","lag_672",
    "pv_lag1","pv_lag96",
    # Rolling
    "roll_4_mean","roll_16_mean","roll_96_mean","roll_4_std","roll_96_std",
    "net_load_lag1","net_load_lag96",
    # Calendar
    "hour","dow","month","day_of_year","is_weekend","is_holiday","tariff_enc",
    "buy_price",
    # Fourier
    "sin_24h","cos_24h","sin_12h","cos_12h","sin_8h","cos_8h",
    "sin_annual","cos_annual",
    # Weather
    "temperature_2m","shortwave_radiation","cloud_cover","relative_humidity_2m",
    "hdd","cdd","temp_lag96","rad_lag96",
]

TARGET = "load_kw"


def get_train_val(df: pd.DataFrame):
    """2024 data: train=all months except Apr+Sep, val=Apr+Sep 2024"""
    train_mask = (df["timestamp"].dt.year == 2024) & \
                 (~df["timestamp"].dt.month.isin([4, 9]))
    val_mask   = (df["timestamp"].dt.year == 2024) & \
                 (df["timestamp"].dt.month.isin([4, 9]))

    df_train = df[train_mask].dropna(subset=FEATURE_COLS + [TARGET])
    df_val   = df[val_mask].dropna(subset=FEATURE_COLS + [TARGET])
    return df_train, df_val


def get_test(df: pd.DataFrame):
    """Apr + Sep 2025 test set"""
    mask = (df["timestamp"].dt.year == 2025) & \
           (df["timestamp"].dt.month.isin([4, 9]))
    return df[mask].dropna(subset=FEATURE_COLS)


if __name__ == "__main__":
    print("Building feature matrix...")
    df = build(use_weather=True)
    print(f"  Total rows    : {len(df)}")

    train, val = get_train_val(df)
    test       = get_test(df)
    print(f"  Train rows    : {len(train)}")
    print(f"  Val rows      : {len(val)}")
    print(f"  Test rows     : {len(test)}")
    print(f"  Feature count : {len(FEATURE_COLS)}")

    df.to_parquet(OUT_DIR / "features_all.parquet", index=False)
    train.to_parquet(OUT_DIR / "features_train.parquet", index=False)
    val.to_parquet(OUT_DIR / "features_val.parquet", index=False)
    test.to_parquet(OUT_DIR / "features_test.parquet", index=False)
    print(f"  Saved to {OUT_DIR}")

    # Quick NaN check
    nan_cols = df[FEATURE_COLS].isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols):
        print("\n  NaN counts (expected for lags at start):")
        print(nan_cols.to_string())
