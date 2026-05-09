"""
Enhanced feature matrix v2 — pushing for sub-50% NRMSE.
Adds:
  - Many more lags (1 through 4 weeks)
  - Recent-change features (acceleration, derivatives)
  - Hour-of-week / hour-of-day mean from TRAIN data only (no leakage)
  - Net-load lags + rolling stats
  - PV cumulative, time-since-PV-onset
  - Interaction features (hour × weekend, hour × season)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROCESSED = ROOT / "data/processed/dataset_processed.csv"
WEATHER   = ROOT / "data/external/sondrio_weather.csv"
OUT_DIR   = ROOT / "data/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY = 16.0
FIXED_HOLIDAYS = {(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)}
EASTER_MONDAYS = {2024:(4,1), 2025:(4,21)}
LOCAL_HOLIDAYS = {(6,19)}


def is_holiday(ts: pd.Series) -> pd.Series:
    out = []
    for t in ts:
        md = (t.month, t.day)
        em = EASTER_MONDAYS.get(t.year, None)
        out.append(int(md in FIXED_HOLIDAYS or md in LOCAL_HOLIDAYS or (em and md == em)))
    return pd.Series(out, index=ts.index)


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t_min = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    for h, lab in [(24, "24h"), (12, "12h"), (8, "8h"), (6, "6h"), (4, "4h")]:
        p = h * 60
        df[f"sin_{lab}"] = np.sin(2 * np.pi * t_min / p)
        df[f"cos_{lab}"] = np.cos(2 * np.pi * t_min / p)
    doy = df["timestamp"].dt.day_of_year
    days = 365.25
    df["sin_annual"] = np.sin(2 * np.pi * doy / days)
    df["cos_annual"] = np.cos(2 * np.pi * doy / days)
    # half-year and quarter-year
    df["sin_semiann"] = np.sin(4 * np.pi * doy / days)
    df["cos_semiann"] = np.cos(4 * np.pi * doy / days)
    return df


def build():
    df = pd.read_csv(PROCESSED, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ── Calendar ──────────────────────────────────────────────────
    df["hour"]  = df["timestamp"].dt.hour
    df["minute"]= df["timestamp"].dt.minute
    df["dow"]   = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.day_of_year
    df["week_of_year"]= df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["is_holiday"]  = is_holiday(df["timestamp"])
    df["tariff_enc"]  = df["tariff_band"].map({"F1": 0, "F2": 1, "F3": 2}).fillna(0).astype(int)

    df = add_fourier(df)

    # Interaction features
    df["hour_x_wknd"]  = df["hour"] * df["is_weekend"]
    df["hour_x_hol"]   = df["hour"] * df["is_holiday"]

    # ── Lags (load) — many more ───────────────────────────────────
    LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192, 288, 384, 480, 576, 672, 1344, 2016]
    for lag in LAGS:
        df[f"lag_{lag}"] = df["load_kw"].shift(lag)

    # Recent changes
    df["d_lag1"]   = df["lag_1"]   - df["lag_2"]
    df["d_lag4"]   = df["lag_4"]   - df["lag_8"]
    df["d_lag96"]  = df["lag_96"]  - df["lag_192"]
    df["d_lag672"] = df["lag_672"] - df["lag_1344"]

    # Lags (PV)
    for lag in [1, 4, 8, 96, 192, 672]:
        df[f"pv_lag{lag}"] = df["pv_kw"].shift(lag)

    # ── Rolling stats (load) ──────────────────────────────────────
    for w in [4, 8, 16, 96, 384, 672]:
        df[f"roll_{w}_mean"] = df["load_kw"].shift(1).rolling(w).mean()
    for w in [4, 16, 96]:
        df[f"roll_{w}_std"]  = df["load_kw"].shift(1).rolling(w).std()
        df[f"roll_{w}_max"]  = df["load_kw"].shift(1).rolling(w).max()
        df[f"roll_{w}_min"]  = df["load_kw"].shift(1).rolling(w).min()

    # ── Net load + rolls ──────────────────────────────────────────
    df["net_load_lag1"]   = df["net_load"].shift(1)
    df["net_load_lag4"]   = df["net_load"].shift(4)
    df["net_load_lag96"]  = df["net_load"].shift(96)
    df["net_load_lag672"] = df["net_load"].shift(672)
    df["net_load_roll96_mean"] = df["net_load"].shift(1).rolling(96).mean()

    # ── Hour-of-week mean from TRAIN ONLY (no leakage) ────────────
    train_mask = (df["timestamp"].dt.year == 2024) & (~df["timestamp"].dt.month.isin([4, 9]))
    train_df = df[train_mask]
    # Quarter-hour-of-week (672 unique slots: 7 days * 96 quarters)
    df["qow"] = df["dow"] * 96 + df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    train_df = train_df.assign(qow=df.loc[train_mask, "qow"].values)
    qow_load_mean = train_df.groupby("qow")["load_kw"].mean()
    qow_load_std  = train_df.groupby("qow")["load_kw"].std()
    qow_load_med  = train_df.groupby("qow")["load_kw"].median()
    df["qow_mean"]   = df["qow"].map(qow_load_mean)
    df["qow_std"]    = df["qow"].map(qow_load_std).fillna(0)
    df["qow_median"] = df["qow"].map(qow_load_med)

    # Holiday-vs-non hour-of-day mean
    train_df = train_df.assign(
        hod=df.loc[train_mask, "hour"].values * 4 + df.loc[train_mask, "minute"].values // 15
    )
    hod_load_hol  = train_df[train_df["is_holiday"] == 1].groupby("hod")["load_kw"].mean()
    hod_load_reg  = train_df[train_df["is_holiday"] == 0].groupby("hod")["load_kw"].mean()
    df["hod"] = df["hour"] * 4 + df["minute"] // 15
    df["hod_mean_holiday"] = df["hod"].map(hod_load_hol).fillna(qow_load_mean.mean())
    df["hod_mean_regular"] = df["hod"].map(hod_load_reg).fillna(qow_load_mean.mean())

    # PV-aware hour-of-day mean (low-PV vs high-PV days)
    daily_pv = df.set_index("timestamp")["pv_kw"].resample("D").sum()
    df["pv_today_total"] = df["timestamp"].dt.normalize().map(daily_pv)
    pv_median = daily_pv.median()
    df["is_high_pv_day"] = (df["pv_today_total"] > pv_median).astype(int)

    # ── Weather ───────────────────────────────────────────────────
    if WEATHER.exists():
        wx = pd.read_csv(WEATHER, parse_dates=["timestamp"])
        wx["timestamp"] = wx["timestamp"].dt.round("15min")
        wx = wx.groupby("timestamp").first().reset_index()
        # Add HDD/CDD if not present
        if "hdd" not in wx.columns and "temperature_2m" in wx.columns:
            wx["hdd"] = (18 - wx["temperature_2m"]).clip(lower=0)
        if "cdd" not in wx.columns and "temperature_2m" in wx.columns:
            wx["cdd"] = (wx["temperature_2m"] - 24).clip(lower=0)
        cols = ["timestamp","temperature_2m","shortwave_radiation","cloud_cover",
                "relative_humidity_2m","hdd","cdd"]
        cols = [c for c in cols if c in wx.columns]
        df = df.merge(wx[cols], on="timestamp", how="left")
        # Weather lags
        if "temperature_2m" in df:
            df["temp_lag1"]    = df["temperature_2m"].shift(1)
            df["temp_lag96"]   = df["temperature_2m"].shift(96)
            df["temp_lag672"]  = df["temperature_2m"].shift(672)
            df["temp_d_lag96"] = df["temperature_2m"] - df["temp_lag96"]
        if "shortwave_radiation" in df:
            df["rad_lag96"]    = df["shortwave_radiation"].shift(96)
            df["rad_roll96_mean"] = df["shortwave_radiation"].rolling(96).mean()
        # Solar position proxy: cos of zenith ~ shortwave_radiation if available
        # Already captured by Fourier, but interaction: temp × is_weekend
        if "temperature_2m" in df:
            df["temp_x_wknd"] = df["temperature_2m"] * df["is_weekend"]
            df["hdd_x_wknd"]  = df.get("hdd", 0)              * df["is_weekend"]

    return df


FEATURE_COLS_V2 = None  # populated below


def get_feature_cols(df):
    """Return all model-ready feature columns (exclude targets/IDs).
    sell_price is excluded — dynamic market price is NOT knowable ahead of time.
    pv_today_total is excluded — leaks future PV from same day."""
    drop = {"timestamp", "load_kw", "p_battery_kw", "grid_kw", "tariff_band", "minute",
            "qow", "hod", "net_load", "sell_price", "pv_today_total"}
    return [c for c in df.columns if c not in drop and not c.startswith("Unnamed")]


def get_train_val_test(df):
    train_mask = (df["timestamp"].dt.year == 2024) & (~df["timestamp"].dt.month.isin([4, 9]))
    val_mask   = (df["timestamp"].dt.year == 2024) & (df["timestamp"].dt.month.isin([4, 9]))
    test_mask  = (df["timestamp"].dt.year == 2025) & (df["timestamp"].dt.month.isin([4, 9]))
    feats = get_feature_cols(df)
    train = df[train_mask].dropna(subset=feats).reset_index(drop=True)
    val   = df[val_mask  ].dropna(subset=feats).reset_index(drop=True)
    test  = df[test_mask ].dropna(subset=feats).reset_index(drop=True)
    return train, val, test, feats


if __name__ == "__main__":
    print("Building enhanced feature set v2...")
    df = build()
    print(f"  Total rows: {len(df)}  cols: {len(df.columns)}")
    train, val, test, feats = get_train_val_test(df)
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"  Features ({len(feats)}): {feats[:8]}...")
    df.to_parquet(OUT_DIR / "features_v2_all.parquet", index=False)
    train.to_parquet(OUT_DIR / "features_v2_train.parquet", index=False)
    val  .to_parquet(OUT_DIR / "features_v2_val.parquet",   index=False)
    test .to_parquet(OUT_DIR / "features_v2_test.parquet",  index=False)
    # Save feature list
    (OUT_DIR / "features_v2_cols.txt").write_text("\n".join(feats))
    print(f"  Saved -> {OUT_DIR}/features_v2_*.parquet")
