import pandas as pd

df = pd.read_csv("data/processed/dataset_processed.csv", parse_dates=["timestamp"])

# Monday Jan 8 2024 (regular weekday)
mon = df[df["timestamp"].dt.date == pd.Timestamp("2024-01-08").date()]
print("=== Monday Jan 8 2024 ===")
for h in [0, 7, 8, 12, 19, 23]:
    row = mon[mon["timestamp"].dt.hour == h].iloc[0]
    print(f"  {row['timestamp']}  band={row['tariff_band']}  buy={row['buy_price']}  holiday={row['is_holiday']}")

# Sunday Jan 7 2024
sun = df[df["timestamp"].dt.date == pd.Timestamp("2024-01-07").date()]
print("\n=== Sunday Jan 7 2024 (should be F3 all day) ===")
row = sun[sun["timestamp"].dt.hour == 12].iloc[0]
print(f"  {row['timestamp']}  band={row['tariff_band']}  buy={row['buy_price']}  holiday={row['is_holiday']}")

# April 25 2024 (Liberation Day - Thursday but holiday -> F3)
h25 = df[df["timestamp"].dt.date == pd.Timestamp("2024-04-25").date()]
print("\n=== April 25 2024 (Liberation Day holiday - should be F3) ===")
row = h25[h25["timestamp"].dt.hour == 10].iloc[0]
print(f"  {row['timestamp']}  band={row['tariff_band']}  buy={row['buy_price']}  holiday={row['is_holiday']}")

# Saturday check
sat = df[df["timestamp"].dt.date == pd.Timestamp("2024-01-06").date()]
print("\n=== Saturday Jan 6 2024 (Epiphany - holiday, should be F3) ===")
row = sat[sat["timestamp"].dt.hour == 10].iloc[0]
print(f"  {row['timestamp']}  band={row['tariff_band']}  buy={row['buy_price']}  holiday={row['is_holiday']}")

# Regular Saturday (Jan 13)
sat2 = df[df["timestamp"].dt.date == pd.Timestamp("2024-01-13").date()]
print("\n=== Regular Saturday Jan 13 2024 ===")
for h in [3, 10, 23]:
    row = sat2[sat2["timestamp"].dt.hour == h].iloc[0]
    print(f"  {row['timestamp']}  band={row['tariff_band']}  buy={row['buy_price']}")

print("\n=== Cost structure ===")
print(f"Mean buy price  : {df['buy_price'].mean():.4f} EUR/kWh")
print(f"Mean sell price : {df['sell_price'].mean():.4f} EUR/kWh")
print(f"Buy/sell ratio  : {df['buy_price'].mean() / df['sell_price'].mean():.2f}x")
print(f"\nBand counts:\n{df['tariff_band'].value_counts().to_string()}")
print(f"\nHoliday timesteps: {df['is_holiday'].sum():,}")

# Quick bill estimate for 2025 using historical battery
test = df[df["timestamp"].dt.year == 2025].copy()
test["cost"] = test.apply(
    lambda r: r["grid_kw"] * r["buy_price"] * 0.25 if r["grid_kw"] > 0
              else abs(r["grid_kw"]) * r["sell_price"] * 0.25 * -1,
    axis=1
)
print(f"\n=== Quick 2025 bill estimate (Baseline A - existing controller) ===")
print(f"Total 2025 bill : EUR {test['cost'].sum():.2f}")
print(f"Import cost     : EUR {test[test['grid_kw']>0]['cost'].sum():.2f}")
print(f"Export revenue  : EUR {-test[test['grid_kw']<0]['cost'].sum():.2f}")
