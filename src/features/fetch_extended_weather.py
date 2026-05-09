"""
Pull EXTENDED weather variables from Open-Meteo (free, no API key).
Beyond the v1 set (temperature, radiation, cloud, humidity), we add:
  - wind_speed_10m, wind_gusts_10m, wind_direction_10m
  - precipitation, rain, snowfall
  - surface_pressure, pressure_msl
  - dew_point_2m, apparent_temperature
  - direct_radiation, diffuse_radiation, direct_normal_irradiance
  - is_day flag (Open-Meteo built-in)
  - sunrise / sunset (daily)
"""
import numpy as np
import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).parents[2]
OUT  = ROOT / "data/external/sondrio_weather_extended.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

LAT, LON = 46.17, 9.87  # Sondrio
START, END = "2024-01-01", "2025-12-31"

HOURLY_VARS = [
    "temperature_2m", "shortwave_radiation", "cloud_cover", "relative_humidity_2m",
    "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
    "precipitation", "rain", "snowfall",
    "surface_pressure", "pressure_msl",
    "dew_point_2m", "apparent_temperature",
    "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
    "is_day",
]
DAILY_VARS = ["sunrise", "sunset", "daylight_duration"]


def fetch():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "start_date": START,
        "end_date":   END,
        "hourly":    ",".join(HOURLY_VARS),
        "daily":     ",".join(DAILY_VARS),
        "timezone":  "Europe/Rome",
        "wind_speed_unit": "ms",
    }
    print(f"Fetching extended weather for Sondrio ({LAT}, {LON}) {START}..{END}")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    j = r.json()

    # Hourly -> dataframe
    h = pd.DataFrame(j["hourly"])
    h["timestamp"] = pd.to_datetime(h["time"])
    h = h.drop(columns=["time"])
    print(f"  hourly: {len(h)} rows, {len(h.columns)-1} vars")

    # Daily features (for sunrise/sunset etc.)
    d = pd.DataFrame(j["daily"])
    d["date"] = pd.to_datetime(d["time"]).dt.date
    d["sunrise"] = pd.to_datetime(d["sunrise"])
    d["sunset"]  = pd.to_datetime(d["sunset"])
    print(f"  daily : {len(d)} rows")

    # Resample hourly -> 15-min, interpolate
    h = h.set_index("timestamp").sort_index()
    h_15 = h.resample("15min").interpolate(method="linear")
    h_15["timestamp"] = h_15.index
    h_15 = h_15.reset_index(drop=True)

    # Merge daily features (broadcast to 15-min)
    h_15["date"] = h_15["timestamp"].dt.date
    daily_use = d[["date", "sunrise", "sunset", "daylight_duration"]]
    out = h_15.merge(daily_use, on="date", how="left")
    out = out.drop(columns=["date"])

    # Derived: hours since sunrise / until sunset
    out["sunrise"] = pd.to_datetime(out["sunrise"])
    out["sunset"]  = pd.to_datetime(out["sunset"])
    out["hrs_since_sunrise"] = (out["timestamp"] - out["sunrise"]).dt.total_seconds() / 3600.0
    out["hrs_until_sunset"]  = (out["sunset"]  - out["timestamp"]).dt.total_seconds() / 3600.0
    # Cap negatives at 0 (before sunrise / after sunset)
    out["hrs_since_sunrise"] = out["hrs_since_sunrise"].clip(lower=0, upper=24)
    out["hrs_until_sunset"]  = out["hrs_until_sunset"].clip(lower=0, upper=24)
    out["daylight_h"] = out["daylight_duration"] / 3600.0

    # HDD/CDD from temperature (Italian residential thresholds)
    out["hdd"] = (18.0 - out["temperature_2m"]).clip(lower=0)
    out["cdd"] = (out["temperature_2m"] - 24.0).clip(lower=0)

    # Drop intermediate
    out = out.drop(columns=["sunrise", "sunset", "daylight_duration"])

    out.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    print(f"  shape: {out.shape}")
    print(f"  columns: {list(out.columns)}")
    return out


if __name__ == "__main__":
    fetch()
