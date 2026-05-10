import requests
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parents[2] / "data" / "external" / "sondrio_weather.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

LAT, LON = 46.17, 9.87

def fetch_weather(start="2024-01-01", end="2025-12-31") -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,shortwave_radiation,cloud_cover,relative_humidity_2m,windspeed_10m,precipitation,snowfall,weather_code",
        "timezone": "Europe/Rome",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def interpolate_to_15min(df_hourly: pd.DataFrame) -> pd.DataFrame:
    df = df_hourly.set_index("timestamp")
    # Create target index
    target_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="15min")
    df_15 = df.reindex(target_idx)
    
    # Numerical columns to interpolate
    num_cols = ["temperature_2m", "shortwave_radiation", "cloud_cover", 
                "relative_humidity_2m", "windspeed_10m", "precipitation", "snowfall"]
    num_cols = [c for c in num_cols if c in df_15.columns]
    df_15[num_cols] = df_15[num_cols].interpolate(method="linear")
    
    # Categorical columns to ffill
    cat_cols = ["weather_code"]
    cat_cols = [c for c in cat_cols if c in df_15.columns]
    df_15[cat_cols] = df_15[cat_cols].ffill()
    
    df_15 = df_15.reset_index().rename(columns={"index": "timestamp"})
    return df_15


def add_hdd_cdd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hdd"] = (18.0 - df["temperature_2m"]).clip(lower=0)
    df["cdd"] = (df["temperature_2m"] - 24.0).clip(lower=0)
    return df


if __name__ == "__main__":
    print("Fetching Sondrio weather 2024-2025 (Hourly Raw)...")
    raw = fetch_weather()
    raw = add_hdd_cdd(raw)
    raw.to_csv(OUT, index=False)
    print(f"  Saved -> {OUT}")

