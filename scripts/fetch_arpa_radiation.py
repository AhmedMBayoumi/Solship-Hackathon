"""
Fetch ARPA Lombardia ground-truth radiation for Sondrio.
Sensor 2098 = Sondrio Fond. Fojanini (46.1676 N, 9.8509 E, 307m)
Dataset cxym-eps2 = "Radiazione Globale dal 2021"

Cadence: 10-min  ->  resample to 15-min (mean) to match load data.
Coverage needed: 2024-01-01 to 2025-09-30
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).parents[1]
OUT = ROOT / "data/external/arpa_sondrio_radiation.csv"

SENSOR_ID = "2098"
DATASET   = "cxym-eps2"
URL       = f"https://www.dati.lombardia.it/resource/{DATASET}.json"

# Pull in chunks of 50k rows (Socrata's max default).  10-min cadence x 24h x 365d
# = ~52k rows/year, so 2 years needs ~3 chunks.
def fetch_range(start, end, chunk_size=50000):
    print(f"  fetching {start} -> {end}")
    rows = []
    offset = 0
    while True:
        params = {
            "idsensore": SENSOR_ID,
            "$where":    f"data between '{start}T00:00:00' and '{end}T00:00:00'",
            "$select":   "data, valore, stato",
            "$order":    "data ASC",
            "$limit":    chunk_size,
            "$offset":   offset,
        }
        r = requests.get(URL, params=params, timeout=120)
        r.raise_for_status()
        batch = r.json()
        if not batch: break
        rows.extend(batch)
        print(f"    pulled {len(batch)} rows (cumulative {len(rows)})")
        if len(batch) < chunk_size: break
        offset += chunk_size
    return rows

print(f"Fetching ARPA sensor {SENSOR_ID} radiation 2024-01-01 to 2025-10-01...")
t0 = time.time()
rows = []
# Fetch year by year to avoid huge requests
for start, end in [
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"),
    ("2025-07-01", "2025-10-01"),
]:
    rows.extend(fetch_range(start, end))
print(f"  total rows: {len(rows)}  ({time.time()-t0:.0f}s)")

df = pd.DataFrame(rows)
df["data"]   = pd.to_datetime(df["data"])
df["valore"] = pd.to_numeric(df["valore"], errors="coerce")
df = df.sort_values("data").reset_index(drop=True)
print(f"  raw 10-min data: {len(df)} rows, range {df['data'].min()} -> {df['data'].max()}")
print(f"  valid stato values: {df['stato'].value_counts().to_dict()}")

# Drop bad-state rows
bad = df["stato"] != "VA"
if bad.sum():
    print(f"  dropping {bad.sum()} non-VA rows")
df = df[~bad].copy()

# Resample to 15-min mean (matching load data cadence)
df = df.set_index("data")
df15 = df["valore"].resample("15min").mean().to_frame("arpa_radiation")
df15.index.name = "timestamp"
df15 = df15.reset_index()

# Sanity check
print(f"  resampled to 15-min: {len(df15)} rows")
print(f"  range {df15['timestamp'].min()} -> {df15['timestamp'].max()}")
print(f"  stats: min={df15['arpa_radiation'].min():.1f}  "
      f"max={df15['arpa_radiation'].max():.1f}  "
      f"mean={df15['arpa_radiation'].mean():.1f}  "
      f"NaN={df15['arpa_radiation'].isna().sum()}")

# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
df15.to_csv(OUT, index=False)
print(f"\nSaved -> {OUT}")
