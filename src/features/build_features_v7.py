"""
Feature set v7 — v6 + CAUSAL wavelet decomposition features.

Wavelet idea (Gemini's recommendation):
  Decompose past 4 days of load into multi-scale frequency bands:
    - L1 (high freq): 15-30 min noise, captures appliance switching
    - L2 (mid freq):  1-2 hour patterns
    - L3 (low freq):  daily-scale trend
    - Approximation:  long-term baseline
  Each band becomes a feature at time t. The model can then learn the
  smooth trend separately from the spike patterns.

Causality:
  At each time t, we compute the wavelet decomposition using ONLY past
  load values (load[t-W : t]). Then we use the last-position value of
  each reconstructed band. NO future leakage.

Implementation: PyWavelets (db4 wavelet, level=3, W=192 = 2 days).
"""
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import pywt
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyWavelets", "-q"])
    import pywt

ROOT = Path(__file__).parents[2]
PROCESSED   = ROOT / "data/processed/dataset_processed.csv"
WEATHER_EXT = ROOT / "data/external/sondrio_weather_extended.csv"
CLEARSKY    = ROOT / "data/external/sondrio_clearsky.csv"
V6_FEATS    = ROOT / "data/features/features_v6_all.parquet"
OUT_DIR     = ROOT / "data/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


WAVELET    = "db4"
LEVEL      = 3
WINDOW     = 192     # 2 days (192 * 15min)
MIN_WINDOW = 32      # need at least 32 past values to compute level-3 wavelet


def causal_wavelet_features(load_series: np.ndarray, wavelet=WAVELET, level=LEVEL, window=WINDOW):
    """
    For each time t, compute wavelet decomposition on load[max(0,t-window):t]
    and return the LAST value of each reconstructed band.

    Returns shape (n, level+1): [L1, L2, ..., L_level, Approx]
    L1 is highest frequency; Approx is smoothest trend.
    """
    n = len(load_series)
    n_bands = level + 1
    out = np.full((n, n_bands), np.nan, dtype="float32")

    for t in range(MIN_WINDOW, n):
        win_start = max(0, t - window)
        win = load_series[win_start:t]
        if len(win) < MIN_WINDOW:
            continue
        # Forward fill any NaN inside the window
        if np.isnan(win).any():
            mask = np.isnan(win)
            if mask.all():
                continue
            # Simple forward fill
            last_v = 0.0
            for i in range(len(win)):
                if np.isnan(win[i]):
                    win[i] = last_v
                else:
                    last_v = win[i]
        # Decompose
        try:
            coeffs = pywt.wavedec(win, wavelet, level=level)
        except Exception:
            continue
        # Reconstruct each band individually to original length
        for i, c in enumerate(coeffs):
            c_only = [np.zeros_like(x) for x in coeffs]
            c_only[i] = c
            try:
                rec = pywt.waverec(c_only, wavelet)[:len(win)]
                out[t, i] = rec[-1]
            except Exception:
                pass
    return out


def build():
    print("Loading v6 features (will extend with wavelet)...", flush=True)
    if V6_FEATS.exists():
        df = pd.read_parquet(V6_FEATS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # Fallback: rebuild from scratch (slower)
        from src.features.build_features_v6 import build as build_v6
        df = build_v6()

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df)} rows, {df.shape[1]} columns", flush=True)

    # Compute causal wavelet on load_kw
    print(f"Computing causal wavelet ({WAVELET}, level={LEVEL}, window={WINDOW})...", flush=True)
    import time
    t0 = time.time()
    wavelet_arr = causal_wavelet_features(df["load_kw"].values.astype("float64"))
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # Add as columns: wavelet_L1, L2, L3, approx
    for i in range(wavelet_arr.shape[1]):
        col = f"wavelet_L{LEVEL - i}" if i < LEVEL else "wavelet_approx"
        df[col] = wavelet_arr[:, i]

    # Add lagged versions (1-step shifted) for safety — at time t, feature uses load[t-1] and earlier
    # But wait, our compute already uses load[t-window:t] so it's already past-only.
    # However, since v6 lag features use load[t-1] anyway, the wavelet at index t is
    # using load up to index t-1. So shift by 0 is correct for prediction at t.
    # To be extra safe, add a shifted-by-1 version too.
    for i in range(wavelet_arr.shape[1]):
        col = f"wavelet_L{LEVEL - i}_lag1" if i < LEVEL else "wavelet_approx_lag1"
        base_col = f"wavelet_L{LEVEL - i}" if i < LEVEL else "wavelet_approx"
        df[col] = df[base_col].shift(1)

    # Multi-scale spike features: ratio of recent high-freq to low-freq
    df["wavelet_hi_lo_ratio"]  = df["wavelet_L1"] / (df["wavelet_approx"].abs() + 0.1)
    df["wavelet_volatility"]   = df["wavelet_L1"].abs() + df["wavelet_L2"].abs()

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
    print("Building v7 features (v6 + causal wavelet)...", flush=True)
    df = build()
    feats = get_feature_cols(df)
    print(f"  Total rows: {len(df)}  cols: {df.shape[1]}  features: {len(feats)}", flush=True)

    df.to_parquet(OUT_DIR / "features_v7_all.parquet", index=False)
    train, val, test, _ = get_train_val_test(df)
    train.to_parquet(OUT_DIR / "features_v7_train.parquet", index=False)
    val  .to_parquet(OUT_DIR / "features_v7_val.parquet",   index=False)
    test .to_parquet(OUT_DIR / "features_v7_test.parquet",  index=False)
    (OUT_DIR / "features_v7_cols.txt").write_text("\n".join(feats))
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}", flush=True)

    v6_set = set((OUT_DIR / "features_v6_cols.txt").read_text().splitlines()) if (OUT_DIR / "features_v6_cols.txt").exists() else set()
    new_feats = [f for f in feats if f not in v6_set]
    print(f"\nNEW wavelet features in v7 ({len(new_feats)}):", flush=True)
    for f in new_feats: print(f"  + {f}")
