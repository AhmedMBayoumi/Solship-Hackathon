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
        "hourly": "temperature_2m,shortwave_radiation,cloud_cover,relative_humidity_2m,windspeed_10m",
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
    df_15 = df.resample("15min").interpolate("linear")
    df_15 = df_15.reset_index()
    return df_15


def add_hdd_cdd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hdd"] = (18.0 - df["temperature_2m"]).clip(lower=0)
    df["cdd"] = (df["temperature_2m"] - 24.0).clip(lower=0)
    return df


if __name__ == "__main__":
    print("Fetching Sondrio weather 2024-2025...")
    raw = fetch_weather()
    print(f"  Raw hourly rows: {len(raw)}")
    df15 = interpolate_to_15min(raw)
    df15 = add_hdd_cdd(df15)
    print(f"  15-min rows    : {len(df15)}")
    print(f"  Columns        : {list(df15.columns)}")
    df15.to_csv(OUT, index=False)
    print(f"  Saved -> {OUT}")
    print(df15.describe().to_string())

