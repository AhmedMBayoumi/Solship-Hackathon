"""
Build processed dataset from raw CSV.
Adds: buy_price (Italian TOU tariff), tariff_band, net_load, sell_price forward-fill.
Renames columns to match the official hackathon spec.
Saves to data/processed/dataset_processed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH  = Path("data/raw/ENERGY_Hackathon_DataSet.csv")
OUT_PATH  = Path("data/processed/dataset_processed.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 1. LOAD RAW ───────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH, sep=";", decimal=",", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df):,} rows from raw CSV.")

# ── 2. RENAME COLUMNS TO MATCH OFFICIAL SPEC ─────────────────────────────────
df = df.rename(columns={
    "load_p":                 "load_kw",
    "pv_p":                   "pv_kw",
    "battery_p":              "p_battery_kw",
    "grid_p":                 "grid_kw",
    "Selling_price_eur_kwh":  "sell_price",
})

# ── 3. FORWARD-FILL 8 MISSING SELL PRICES (DST GAPS) ─────────────────────────
n_missing_before = df["sell_price"].isna().sum()
df["sell_price"] = df["sell_price"].ffill()
print(f"Forward-filled {n_missing_before} missing sell_price values.")

# ── 4. ITALIAN NATIONAL HOLIDAYS ─────────────────────────────────────────────
# Fixed holidays (same every year)
FIXED_HOLIDAYS = [(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)]

# Easter dates (pre-computed for 2024 and 2025)
# Easter 2024: March 31  → Easter Monday April 1
# Easter 2025: April 20  → Easter Monday April 21
EASTER_MONDAYS = {2024: (4,1), 2025: (4,21)}

def is_italian_holiday(dt):
    month, day, year = dt.month, dt.day, dt.year
    if (month, day) in FIXED_HOLIDAYS:
        return True
    em = EASTER_MONDAYS.get(year)
    if em and (month, day) == em:
        return True
    return False

holidays = df["timestamp"].apply(is_italian_holiday)
print(f"Italian holiday timesteps: {holidays.sum():,}")

# ── 5. COMPUTE TARIFF BAND ────────────────────────────────────────────────────
# F1 (Peak)     : Mon–Fri 08:00–19:00 (excl. holidays)         → €0.2540/kWh
# F2 (Shoulder) : Mon–Fri 07:00–08:00 & 19:00–23:00 (excl. holidays)
#                 Saturday 07:00–23:00 (excl. holidays)         → €0.2682/kWh
# F3 (Off-peak) : Mon–Sat 00:00–07:00 & 23:00–24:00 (excl. holidays)
#                 Sundays and all national holidays (all day)   → €0.2440/kWh

def tariff_band(row):
    ts      = row["timestamp"]
    holiday = row["is_holiday"]
    dow     = ts.dayofweek   # 0=Mon, 6=Sun
    hour    = ts.hour        # 0–23 (interval start)

    # Sundays or holidays → F3 all day
    if dow == 6 or holiday:
        return "F3"

    # Saturday
    if dow == 5:
        if 7 <= hour < 23:
            return "F2"
        return "F3"   # Sat 00–07 and 23–24

    # Monday–Friday
    if 8 <= hour < 19:
        return "F1"
    if (7 <= hour < 8) or (19 <= hour < 23):
        return "F2"
    return "F3"   # 00–07 and 23–24

TARIFF_PRICES = {"F1": 0.2540, "F2": 0.2682, "F3": 0.2440}

df["is_holiday"]   = holidays
df["tariff_band"]  = df.apply(tariff_band, axis=1)
df["buy_price"]    = df["tariff_band"].map(TARIFF_PRICES)

print("\nTariff band distribution:")
print(df["tariff_band"].value_counts().to_string())

# ── 6. DERIVED FEATURES ───────────────────────────────────────────────────────
df["net_load"] = df["load_kw"] - df["pv_kw"]   # positive = need grid/battery

# ── 7. REORDER COLUMNS ───────────────────────────────────────────────────────
cols = [
    "timestamp",
    "load_kw",
    "pv_kw",
    "p_battery_kw",
    "grid_kw",
    "buy_price",
    "sell_price",
    "tariff_band",
    "is_holiday",
    "net_load",
]
df = df[cols]

# ── 8. SAVE ───────────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved processed dataset to {OUT_PATH}")
print(f"Shape: {df.shape}")
print("\nFirst 3 rows:")
print(df.head(3).to_string())
print("\nDescribe buy_price:")
print(df["buy_price"].describe().to_string())
print("\nMissing values in processed:")
print(df.isna().sum().to_string())
