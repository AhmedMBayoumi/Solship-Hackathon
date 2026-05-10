"""
PHANN-style physical baseline: compute deterministic clear-sky GHI/DNI/DHI
+ solar position for Sondrio, then ratio features (transparency / clearness).

The model only has to learn the WEATHER DEVIATION from clear-sky, not the
entire physics of the sun's orbit. This often unlocks several percentage
points on irradiance-driven targets (PV) and indirectly on load via
HVAC/lighting timing.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import pvlib
from pvlib.location import Location

ROOT = Path(__file__).parents[2]
OUT  = ROOT / "data/external/sondrio_clearsky.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

LAT, LON, ALT, TZ = 46.17, 9.87, 307.0, "Europe/Rome"
START, END = "2024-01-01 00:00", "2025-12-31 23:45"


def fetch():
    print(f"Computing clear-sky GHI/DNI/DHI for Sondrio ({LAT}, {LON}, {ALT}m)")
    loc = Location(LAT, LON, TZ, ALT, "Sondrio")
    times = pd.date_range(START, END, freq="15min", tz=TZ)

    cs = loc.get_clearsky(times, model="ineichen")            # ghi, dni, dhi
    sp = loc.get_solarposition(times)                          # zenith, azimuth, elevation
    out = cs.copy()
    out["solar_zenith"]   = sp["apparent_zenith"]
    out["solar_elevation"]= 90 - sp["apparent_zenith"]
    out["solar_azimuth"]  = sp["azimuth"]
    out["clearsky_ghi"]   = cs["ghi"]
    out["clearsky_dni"]   = cs["dni"]
    out["clearsky_dhi"]   = cs["dhi"]
    out = out.drop(columns=["ghi","dni","dhi"])

    # Cosine of zenith — directly proportional to potential PV generation
    z_rad = np.deg2rad(sp["apparent_zenith"].clip(upper=90))
    out["cos_solar_zenith"] = np.cos(z_rad).where(sp["apparent_zenith"] < 90, 0.0)
    out["is_daytime_phys"]  = (out["solar_elevation"] > 0).astype(int)

    # Strip timezone (timestamps in dataset are in local time but tz-naive)
    out["timestamp"] = times.tz_localize(None)
    out = out.reset_index(drop=True)

    out.to_csv(OUT, index=False)
    print(f"  shape: {out.shape}")
    print(f"  cols : {list(out.columns)}")
    print(f"  saved -> {OUT}")
    return out


if __name__ == "__main__":
    fetch()
