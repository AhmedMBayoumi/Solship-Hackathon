"""
Feature builder for the 2nd surprise dataset.
Source: 2nd DataSet.xlsx with cols [timestamp, pv_p, load_p].
Same site (Sondrio per supervisor), so same Italian holiday calendar &
clear-sky physics location apply. NO tariff/price features per supervisor.

Output: features_surprise_all.parquet  (train+test, single parquet file)

Train: 2024-11-25 .. 2026-02-28
Test : 2026-03-01 .. 2026-03-31
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].parent))

import numpy as np
import pandas as pd

try:
    import pywt
except ImportError:
    pywt = None

ROOT = Path(__file__).parents[2]
SRC  = ROOT / "2nd DataSet.xlsx"
OUT  = ROOT / "data/features"
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Load + rename ─────────────────────────────────────────────────
print("Loading 2nd DataSet.xlsx ...", flush=True)
df = pd.read_excel(SRC, sheet_name="Power Data")
df = df.rename(columns={"load_p": "load_kw", "pv_p": "pv_kw"})
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
# clip negative pv (sensor noise)
df["pv_kw"] = df["pv_kw"].clip(lower=0)
df["load_kw"] = df["load_kw"].clip(lower=0)
print(f"  rows={len(df)}  range {df['timestamp'].min()} -> {df['timestamp'].max()}", flush=True)

# ── 2. Calendar features ─────────────────────────────────────────────
ts = df["timestamp"]
df["hour"]         = ts.dt.hour
df["dow"]          = ts.dt.dayofweek
df["month"]        = ts.dt.month
df["day_of_year"]  = ts.dt.dayofyear
df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
df["is_weekend"]   = (df["dow"] >= 5).astype(int)
df["minute"]       = ts.dt.minute
df["qow"]          = (df["dow"] * 96 + df["hour"] * 4 + df["minute"] // 15)   # quarter-of-week
df["hod"]          = df["hour"] * 4 + df["minute"] // 15                       # quarter-of-day

# Italian national holidays (covers 2024-11 .. 2026-03)
IT_HOLS = pd.to_datetime([
    "2024-12-25","2024-12-26","2024-12-08","2024-11-01",
    "2025-01-01","2025-01-06","2025-04-21","2025-04-25","2025-05-01",
    "2025-06-02","2025-08-15","2025-11-01","2025-12-08","2025-12-25","2025-12-26",
    "2026-01-01","2026-01-06","2026-04-06","2026-04-25","2026-05-01",
])
df["is_holiday"] = df["timestamp"].dt.normalize().isin(IT_HOLS).astype(int)
# bridge day = working day adjacent to a holiday
hd = df["timestamp"].dt.normalize()
df["is_bridge_day"] = (
    (hd.shift(-96).isin(IT_HOLS) & ~df["is_holiday"].astype(bool) & ~df["is_weekend"].astype(bool)) |
    (hd.shift(96).isin(IT_HOLS)  & ~df["is_holiday"].astype(bool) & ~df["is_weekend"].astype(bool))
).astype(int)

# ── 3. Cyclical encodings ─────────────────────────────────────────────
for period, label in [(96, "24h"), (48, "12h"), (32, "8h"), (24, "6h"), (16, "4h")]:
    df[f"sin_{label}"] = np.sin(2*np.pi*df["hod"]/period)
    df[f"cos_{label}"] = np.cos(2*np.pi*df["hod"]/period)
df["sin_annual"]  = np.sin(2*np.pi*df["day_of_year"]/365.25)
df["cos_annual"]  = np.cos(2*np.pi*df["day_of_year"]/365.25)
df["sin_semiann"] = np.sin(4*np.pi*df["day_of_year"]/365.25)
df["cos_semiann"] = np.cos(4*np.pi*df["day_of_year"]/365.25)

# Italian lifestyle hour blocks
df["it_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] < 9)).astype(int)
df["it_lunch_peak"]   = ((df["hour"] >= 12) & (df["hour"] < 14)).astype(int)
df["it_tou_shift"]    = ((df["hour"] >= 18) & (df["hour"] < 20)).astype(int)
df["it_dinner_hour"]  = ((df["hour"] >= 19) & (df["hour"] < 22)).astype(int)
df["it_pre_dawn"]     = ((df["hour"] >= 1) & (df["hour"] < 5)).astype(int)

# ── 4. Lag features on load + pv ─────────────────────────────────────
LAG_STEPS = [1,2,3,4,6,8,12,16,24,32,48,64,96,192,288,384,480,576,672,1344,2016]
for lag in LAG_STEPS:
    df[f"lag_{lag}"] = df["load_kw"].shift(lag)
df["d_lag1"]   = df["lag_1"]   - df["lag_2"]
df["d_lag4"]   = df["lag_4"]   - df["lag_8"]
df["d_lag96"]  = df["lag_96"]  - df["lag_192"]
df["d_lag672"] = df["lag_672"] - df["lag_1344"]

for lag in [1, 4, 8, 96, 192, 672]:
    df[f"pv_lag{lag}"] = df["pv_kw"].shift(lag)

# ── 5. Rolling stats ─────────────────────────────────────────────────
for w in [4, 8, 16, 96, 384, 672]:
    df[f"roll_{w}_mean"] = df["load_kw"].shift(1).rolling(w, min_periods=1).mean()
for w in [4, 16, 96]:
    df[f"roll_{w}_std"]  = df["load_kw"].shift(1).rolling(w, min_periods=1).std().fillna(0)
    df[f"roll_{w}_max"]  = df["load_kw"].shift(1).rolling(w, min_periods=1).max()
    df[f"roll_{w}_min"]  = df["load_kw"].shift(1).rolling(w, min_periods=1).min()

# pv-aware net load
df["net_load"]            = (df["load_kw"] - df["pv_kw"]).clip(lower=0)
df["net_load_lag1"]       = df["net_load"].shift(1)
df["net_load_lag4"]       = df["net_load"].shift(4)
df["net_load_lag96"]      = df["net_load"].shift(96)
df["net_load_lag672"]     = df["net_load"].shift(672)
df["net_load_roll96_mean"]= df["net_load"].shift(1).rolling(96, min_periods=1).mean()

# baseload features
df["baseload_24h"]    = df["load_kw"].shift(1).rolling(96,  min_periods=1).quantile(0.10)
df["baseload_1week"]  = df["load_kw"].shift(1).rolling(672, min_periods=1).quantile(0.10)
df["load_range_4h"]   = df["roll_16_max"] - df["roll_16_min"]
df["load_range_24h"]  = df["roll_96_max"] - df["roll_96_min"]
df["is_high_state"]   = (df["lag_1"] > df["roll_96_mean"]).astype(int)
df["is_climbing"]     = (df["d_lag1"] > 0).astype(int)
df["lagged_var_1h"]   = df["roll_4_std"]
df["lagged_var_2h"]   = df["roll_16_std"]

# entropy proxies (smoothed local std)
df["load_entropy_8"]  = df["load_kw"].shift(1).rolling(8,  min_periods=1).std().fillna(0)
df["load_entropy_16"] = df["load_kw"].shift(1).rolling(16, min_periods=1).std().fillna(0)

# pv ratio
df["load_vs_base"]    = df["lag_1"] / (df["baseload_24h"] + 0.01)
df["pv_vs_max"]       = df["pv_kw"] / (df["pv_kw"].shift(1).rolling(96, min_periods=1).max() + 0.01)
df["is_empty_proxy"]  = ((df["lag_1"] < df["baseload_24h"] * 1.2) & (df["hour"].between(8, 18))).astype(int)

# ── 6. Causal wavelet (high-leverage feature) ────────────────────────
if pywt is not None:
    print("Computing causal wavelet features...", flush=True)
    WAVELET, LEVEL, WIN, MIN_W = "db4", 3, 192, 32
    arr = df["load_kw"].values.astype("float64")
    out = np.full((len(arr), LEVEL+1), np.nan, dtype="float32")
    for t in range(MIN_W, len(arr)):
        win_start = max(0, t - WIN)
        win = arr[win_start:t]
        if len(win) < MIN_W: continue
        if np.isnan(win).any():
            mask = np.isnan(win)
            if mask.all(): continue
            last_v = 0.0
            for i in range(len(win)):
                if np.isnan(win[i]):
                    win[i] = last_v
                else:
                    last_v = win[i]
        try:
            coeffs = pywt.wavedec(win, WAVELET, level=LEVEL)
            for i, c in enumerate(coeffs):
                c_only = [np.zeros_like(x) for x in coeffs]
                c_only[i] = c
                rec = pywt.waverec(c_only, WAVELET)[:len(win)]
                out[t, i] = rec[-1]
        except Exception:
            pass
    for i in range(out.shape[1]):
        col = f"wavelet_L{LEVEL - i}" if i < LEVEL else "wavelet_approx"
        df[col] = out[:, i]
    for i in range(out.shape[1]):
        col_lag = f"wavelet_L{LEVEL - i}_lag1" if i < LEVEL else "wavelet_approx_lag1"
        base    = f"wavelet_L{LEVEL - i}"      if i < LEVEL else "wavelet_approx"
        df[col_lag] = df[base].shift(1)
    df["wavelet_hi_lo_ratio"] = df["wavelet_L1"] / (df["wavelet_approx"].abs() + 0.1)
    df["wavelet_volatility"]  = df["wavelet_L1"].abs() + df["wavelet_L2"].abs()
    print(f"  wavelet features added", flush=True)

# ── Save ─────────────────────────────────────────────────────────────
def get_feature_cols(df):
    drop = {"timestamp","load_kw","pv_kw","minute","qow","hod","net_load"}
    return [c for c in df.columns if c not in drop and not c.startswith("Unnamed")]

feats = get_feature_cols(df)
print(f"\nFinal feature count: {len(feats)}", flush=True)
df.to_parquet(OUT / "features_surprise_all.parquet", index=False)
(OUT / "features_surprise_cols.txt").write_text("\n".join(feats))

# Train/test split
test_mask  = (df["timestamp"].dt.year == 2026) & (df["timestamp"].dt.month == 3)
train_mask = ~test_mask
train = df[train_mask].dropna(subset=feats).reset_index(drop=True)
test  = df[test_mask].dropna(subset=feats).reset_index(drop=True)
train.to_parquet(OUT / "features_surprise_train.parquet", index=False)
test .to_parquet(OUT / "features_surprise_test.parquet",  index=False)
print(f"  train rows: {len(train)}  test rows: {len(test)}", flush=True)
print(f"  saved -> {OUT/'features_surprise_all.parquet'}", flush=True)
