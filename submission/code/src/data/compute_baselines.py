import pandas as pd

df = pd.read_csv("data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df25 = df[df["timestamp"].dt.year == 2025].copy()

print("=" * 55)
print("BASELINE A — Existing on-site controller (actual p_battery_kw)")
print("=" * 55)
total_a = 0
for month, name in [(4, "April"), (9, "September")]:
    m = df25[df25["timestamp"].dt.month == month].copy()
    imp = m[m["grid_kw"] > 0].apply(lambda r: r["grid_kw"] * r["buy_price"] * 0.25, axis=1).sum()
    exp = m[m["grid_kw"] < 0].apply(lambda r: abs(r["grid_kw"]) * r["sell_price"] * 0.25, axis=1).sum()
    net = imp - exp
    total_a += net
    print(f"\n  {name} 2025:")
    print(f"    Timesteps     : {len(m)}")
    print(f"    Import cost   : EUR {imp:.2f}")
    print(f"    Export revenue: EUR {exp:.2f}")
    print(f"    Net bill      : EUR {net:.2f}")

print(f"\n  TOTAL Apr+Sep Baseline A : EUR {total_a:.2f}")

print("\n" + "=" * 55)
print("BASELINE B — Zero intelligence (no battery, PV first)")
print("=" * 55)
total_b = 0
for month, name in [(4, "April"), (9, "September")]:
    m = df25[df25["timestamp"].dt.month == month].copy()
    net_load = m["load_kw"] - m["pv_kw"]
    imp = net_load[net_load > 0].mul(m.loc[net_load > 0, "buy_price"]).mul(0.25).sum()
    exp = (-net_load[net_load < 0]).mul(m.loc[net_load < 0, "sell_price"]).mul(0.25).sum()
    net = imp - exp
    total_b += net
    print(f"\n  {name} 2025:")
    print(f"    Import cost   : EUR {imp:.2f}")
    print(f"    Export revenue: EUR {exp:.2f}")
    print(f"    Net bill      : EUR {net:.2f}")

print(f"\n  TOTAL Apr+Sep Baseline B : EUR {total_b:.2f}")
print(f"\n  Baseline A vs B difference: EUR {total_b - total_a:.2f}")
print(f"  (positive = existing controller is better than no battery)")
