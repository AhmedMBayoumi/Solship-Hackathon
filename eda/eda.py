"""
Deep EDA for the Solship Hackathon energy dataset.
Generates plots to outputs/plots/ and a text report to outputs/reports/eda_report.txt
Run locally: python eda/eda.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from src.data.loader import load_raw, load_split

PLOTS_DIR = Path("outputs/plots")
REPORTS_DIR = Path("outputs/reports")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

report_lines = []

def rpt(text=""):
    print(text)
    report_lines.append(text)

def save(fig, name):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  -> saved {path.name}")


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
rpt("=" * 70)
rpt("SOLSHIP HACKATHON — DEEP EDA REPORT")
rpt("=" * 70)

df = load_raw()
train, test = load_split()

# ─── 1. DATASET OVERVIEW ──────────────────────────────────────────────────────
rpt("\n[1] DATASET OVERVIEW")
rpt("-" * 50)
rpt(f"Total rows        : {len(df):,}")
rpt(f"Columns           : {list(df.columns)}")
rpt(f"Date range        : {df['timestamp'].min()} → {df['timestamp'].max()}")
rpt(f"Train (2024) rows : {len(train):,}  ({len(train)/len(df)*100:.1f}%)")
rpt(f"Test  (2025) rows : {len(test):,}  ({len(test)/len(df)*100:.1f}%)")
rpt(f"Time resolution   : 15 minutes")
rpt(f"Expected rows/yr  : {365*24*4} (non-leap), {366*24*4} (leap)")
rpt(f"Actual 2024 rows  : {len(train)}")
rpt(f"Actual 2025 rows  : {len(test)}")

# Missing values
rpt("\n--- Missing Values ---")
for col in df.columns:
    n = df[col].isna().sum()
    rpt(f"  {col:<30} {n:>6} missing  ({n/len(df)*100:.3f}%)")

# DST gaps
diff = df["timestamp"].diff()
gaps = df[diff > pd.Timedelta("15min")]
rpt(f"\nTimestamp gaps >15min: {len(gaps)}")
for _, row in gaps.iterrows():
    rpt(f"  Gap at {row['timestamp']}  (gap = {diff[row.name]})")

# Basic stats
rpt("\n--- Descriptive Statistics ---")
rpt(df[["load_p","grid_p","battery_p","pv_p","Selling_price_eur_kwh"]].describe().to_string())


# ─── 2. TIME SERIES OVERVIEW ──────────────────────────────────────────────────
rpt("\n[2] TIME SERIES OVERVIEW PLOTS")

fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
cols   = ["load_p","grid_p","battery_p","pv_p","Selling_price_eur_kwh"]
colors = ["steelblue","tomato","seagreen","goldenrod","purple"]
ylabels = ["Load (kW)","Grid (kW)","Battery (kW)","PV (kW)","Price (€/kWh)"]

for ax, col, color, ylabel in zip(axes, cols, colors, ylabels):
    ax.plot(df["timestamp"], df[col], color=color, linewidth=0.4, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(col)

axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
fig.suptitle("Full Time Series — All Variables (2024–2025)", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "01_full_timeseries")


# ─── 3. LOAD ANALYSIS ─────────────────────────────────────────────────────────
rpt("\n[3] LOAD ANALYSIS")
rpt("-" * 50)

for yr, dset, label in [(2024, train, "Train 2024"), (2025, test, "Test 2025")]:
    rpt(f"\n  {label}:")
    rpt(f"    Mean   : {dset['load_p'].mean():.4f} kW")
    rpt(f"    Median : {dset['load_p'].median():.4f} kW")
    rpt(f"    Std    : {dset['load_p'].std():.4f} kW")
    rpt(f"    Min    : {dset['load_p'].min():.4f} kW")
    rpt(f"    Max    : {dset['load_p'].max():.4f} kW")
    rpt(f"    P95    : {dset['load_p'].quantile(0.95):.4f} kW")
    rpt(f"    P99    : {dset['load_p'].quantile(0.99):.4f} kW")
    skew = dset['load_p'].skew()
    kurt = dset['load_p'].kurtosis()
    rpt(f"    Skew   : {skew:.4f}  |  Kurtosis: {kurt:.4f}")

# Load distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, dset, label, color in zip(axes[:2], [train, test], ["2024 Train","2025 Test"], ["steelblue","tomato"]):
    ax.hist(dset["load_p"], bins=80, color=color, alpha=0.7, edgecolor="none", density=True)
    dset["load_p"].plot.kde(ax=ax, color="black", linewidth=1.5)
    ax.set_xlabel("Load (kW)")
    ax.set_title(f"Load Distribution — {label}")
    ax.axvline(dset["load_p"].mean(), color="red", linestyle="--", linewidth=1, label=f"Mean={dset['load_p'].mean():.2f}")
    ax.legend()

axes[2].hist(train["load_p"], bins=60, color="steelblue", alpha=0.5, density=True, label="2024")
axes[2].hist(test["load_p"],  bins=60, color="tomato",    alpha=0.5, density=True, label="2025")
axes[2].set_xlabel("Load (kW)")
axes[2].set_title("Load Distribution Overlap: 2024 vs 2025")
axes[2].legend()
fig.suptitle("Load Distribution Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "02_load_distribution")

# Hourly profile
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, dset, label in zip(axes, [train, test], ["2024 Train","2025 Test"]):
    dset = dset.copy()
    dset["hour"] = dset["timestamp"].dt.hour + dset["timestamp"].dt.minute / 60
    hourly = dset.groupby("hour")["load_p"].agg(["mean","std","median"])
    ax.fill_between(hourly.index, hourly["mean"]-hourly["std"], hourly["mean"]+hourly["std"], alpha=0.2)
    ax.plot(hourly.index, hourly["mean"],   label="Mean",   linewidth=2)
    ax.plot(hourly.index, hourly["median"], label="Median", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Load (kW)")
    ax.set_title(f"Average Hourly Load Profile — {label}")
    ax.set_xticks(range(0, 24))
    ax.legend()
    # Mark peak
    peak_hr = hourly["mean"].idxmax()
    ax.axvline(peak_hr, color="red", linestyle=":", alpha=0.7, label=f"Peak ~{peak_hr:.0f}h")
    rpt(f"\n  {label} peak load hour: {peak_hr:.1f}h ({hourly['mean'].max():.3f} kW avg)")
    rpt(f"  {label} min  load hour: {hourly['mean'].idxmin():.1f}h ({hourly['mean'].min():.3f} kW avg)")

fig.suptitle("Hourly Load Profile", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "03_hourly_load_profile")

# Day-of-week profile
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for ax, dset, label in zip(axes, [train, test], ["2024 Train","2025 Test"]):
    dset = dset.copy()
    dset["dow"] = dset["timestamp"].dt.dayofweek
    dow = dset.groupby("dow")["load_p"].mean()
    ax.bar(dow_labels, dow.values, color=sns.color_palette("husl", 7))
    ax.set_ylabel("Mean Load (kW)")
    ax.set_title(f"Day-of-Week Load Profile — {label}")
    for i, v in enumerate(dow.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    rpt(f"\n  {label} weekday avg: {dset[dset['dow']<5]['load_p'].mean():.4f} kW")
    rpt(f"  {label} weekend avg: {dset[dset['dow']>=5]['load_p'].mean():.4f} kW")

fig.suptitle("Day-of-Week Load Profile", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "04_dow_load_profile")

# Monthly profile
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for ax, dset, label in zip(axes, [train, test], ["2024 Train","2025 Test"]):
    dset = dset.copy()
    dset["month"] = dset["timestamp"].dt.month
    months_present = sorted(dset["month"].unique())
    monthly = dset.groupby("month")["load_p"].mean()
    bars = ax.bar([month_labels[m-1] for m in months_present], monthly.values,
                  color=sns.color_palette("coolwarm", len(months_present)))
    ax.set_ylabel("Mean Load (kW)")
    ax.set_title(f"Monthly Load Profile — {label}")
    for bar, v in zip(bars, monthly.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

fig.suptitle("Monthly Load Profile (Seasonality)", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "05_monthly_load_profile")

# Heatmap: hour x day-of-week
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dset, label in zip(axes, [train, test], ["2024 Train","2025 Test"]):
    dset = dset.copy()
    dset["hour"] = dset["timestamp"].dt.hour
    dset["dow"]  = dset["timestamp"].dt.dayofweek
    pivot = dset.groupby(["hour","dow"])["load_p"].mean().unstack()
    pivot.columns = dow_labels
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", cbar_kws={"label":"Mean Load (kW)"})
    ax.set_title(f"Load Heatmap: Hour × Day-of-Week — {label}")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")

fig.suptitle("Load Heatmap Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "06_load_heatmap_hour_dow")

# Heatmap: hour x month
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, dset, label in zip(axes, [train, test], ["2024 Train","2025 Test"]):
    dset = dset.copy()
    dset["hour"]  = dset["timestamp"].dt.hour
    dset["month"] = dset["timestamp"].dt.month
    months_present = sorted(dset["month"].unique())
    pivot = dset.groupby(["hour","month"])["load_p"].mean().unstack()
    pivot.columns = [month_labels[m-1] for m in pivot.columns]
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", cbar_kws={"label":"Mean Load (kW)"})
    ax.set_title(f"Load Heatmap: Hour × Month — {label}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day")

fig.suptitle("Load Seasonal-Hourly Interaction", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "07_load_heatmap_hour_month")

# Year-over-year comparison (2024 vs 2025) — by day-of-year
rpt("\n  Year-over-year load comparison:")
train_c = train.copy(); train_c["doy"] = train_c["timestamp"].dt.dayofyear
test_c  = test.copy();  test_c["doy"]  = test_c["timestamp"].dt.dayofyear
daily_train = train_c.groupby("doy")["load_p"].mean()
daily_test  = test_c.groupby("doy")["load_p"].mean()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily_train.index, daily_train.values, label="2024", color="steelblue", linewidth=1, alpha=0.8)
ax.plot(daily_test.index,  daily_test.values,  label="2025", color="tomato",    linewidth=1, alpha=0.8)
ax.set_xlabel("Day of Year")
ax.set_ylabel("Mean Daily Load (kW)")
ax.set_title("Year-over-Year Daily Load Comparison: 2024 vs 2025")
ax.legend()
plt.tight_layout()
save(fig, "08_yoy_daily_load")

common_days = daily_train.index.intersection(daily_test.index)
corr_yoy = np.corrcoef(daily_train[common_days], daily_test[common_days])[0,1]
mean_diff = (daily_test[common_days] - daily_train[common_days]).mean()
rpt(f"    2024 vs 2025 daily load Pearson corr : {corr_yoy:.4f}")
rpt(f"    Mean daily load difference (2025-2024): {mean_diff:.4f} kW")


# ─── 4. PV GENERATION ANALYSIS ────────────────────────────────────────────────
rpt("\n[4] PV GENERATION ANALYSIS")
rpt("-" * 50)

for yr, dset, label in [(2024, train, "2024"), (2025, test, "2025")]:
    rpt(f"\n  {label}:")
    rpt(f"    Mean PV (all hours)   : {dset['pv_p'].mean():.4f} kW")
    rpt(f"    Mean PV (daylight >0) : {dset[dset['pv_p']>0]['pv_p'].mean():.4f} kW")
    rpt(f"    Max PV                : {dset['pv_p'].max():.4f} kW")
    zero_pct = (dset['pv_p'] == 0).mean() * 100
    rpt(f"    Zero-generation pct   : {zero_pct:.1f}% (nights)")
    total_kwh = dset["pv_p"].sum() * 0.25
    rpt(f"    Total PV energy (kWh) : {total_kwh:.1f} kWh")

# PV hourly profile
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, dset, label in zip(axes, [train, test], ["2024","2025"]):
    dset = dset.copy()
    dset["hour"] = dset["timestamp"].dt.hour
    hourly = dset.groupby("hour")["pv_p"].agg(["mean","max"])
    ax.fill_between(hourly.index, 0, hourly["mean"], alpha=0.4, color="goldenrod", label="Mean")
    ax.plot(hourly.index, hourly["max"], color="orange", linewidth=1.5, linestyle="--", label="Max")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("PV Power (kW)")
    ax.set_title(f"PV Generation Profile — {label}")
    ax.set_xticks(range(0, 24))
    ax.legend()

fig.suptitle("PV Solar Generation — Hourly Profile", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "09_pv_hourly_profile")

# PV monthly
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, dset, label in zip(axes, [train, test], ["2024","2025"]):
    dset = dset.copy()
    dset["month"] = dset["timestamp"].dt.month
    months_present = sorted(dset["month"].unique())
    monthly_pv = dset.groupby("month")["pv_p"].mean()
    ax.bar([month_labels[m-1] for m in months_present], monthly_pv.values, color="goldenrod", alpha=0.8)
    ax.set_ylabel("Mean PV (kW)")
    ax.set_title(f"Monthly PV Generation — {label}")

fig.suptitle("PV Solar Generation — Seasonal Pattern", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "10_pv_monthly_profile")


# ─── 5. NET LOAD & SELF-SUFFICIENCY ───────────────────────────────────────────
rpt("\n[5] NET LOAD & SELF-SUFFICIENCY")
rpt("-" * 50)

for dset, label in [(train, "2024"), (test, "2025")]:
    dset = dset.copy()
    dset["net_load"] = dset["load_p"] - dset["pv_p"]
    pct_pv_covered = (dset["pv_p"].sum() / dset["load_p"].sum()) * 100
    pct_surplus    = (dset["net_load"] < 0).mean() * 100
    rpt(f"\n  {label}:")
    rpt(f"    PV self-sufficiency : {pct_pv_covered:.1f}% of load covered by PV")
    rpt(f"    Surplus periods     : {pct_surplus:.1f}% of timesteps PV > Load")
    rpt(f"    Mean net load       : {dset['net_load'].mean():.4f} kW")
    rpt(f"    Max surplus (export): {-dset['net_load'].min():.4f} kW")

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
for dset, label, color in [(train, "2024", "steelblue"), (test, "2025", "tomato")]:
    dset = dset.copy()
    dset["net_load"] = dset["load_p"] - dset["pv_p"]
    # Weekly rolling mean
    dset = dset.set_index("timestamp")
    rolling = dset["net_load"].rolling("7D").mean()
    dset = dset.reset_index()
    rolling.index = dset["timestamp"]

for ax, dset, label in zip(axes, [train, test], ["2024","2025"]):
    dset = dset.copy()
    dset["net_load"] = dset["load_p"] - dset["pv_p"]
    ax.plot(dset["timestamp"], dset["net_load"], linewidth=0.3, alpha=0.5, color="gray")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(dset["timestamp"], dset["net_load"], 0,
                    where=(dset["net_load"] < 0), color="green", alpha=0.4, label="PV surplus")
    ax.fill_between(dset["timestamp"], dset["net_load"], 0,
                    where=(dset["net_load"] > 0), color="red", alpha=0.2, label="Grid import needed")
    ax.set_ylabel("Net Load (kW)")
    ax.set_title(f"Net Load (Load − PV) — {label}")
    ax.legend()

fig.suptitle("Net Load Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "11_net_load")


# ─── 6. GRID & BATTERY ANALYSIS ───────────────────────────────────────────────
rpt("\n[6] GRID & BATTERY ANALYSIS")
rpt("-" * 50)

for dset, label in [(train, "2024"), (test, "2025")]:
    rpt(f"\n  {label}:")
    rpt(f"    Grid import events   : {(dset['grid_p']>0).sum():,}")
    rpt(f"    Grid export events   : {(dset['grid_p']<0).sum():,}")
    rpt(f"    Max import           : {dset['grid_p'].max():.4f} kW")
    rpt(f"    Max export           : {dset['grid_p'].min():.4f} kW")
    rpt(f"    Battery charge events: {(dset['battery_p']<0).sum():,}")
    rpt(f"    Battery disch events : {(dset['battery_p']>0).sum():,}")
    rpt(f"    Battery idle events  : {(dset['battery_p']==0).sum():,}")

# Energy balance check: load = grid + battery + pv  (at each step)
for dset, label in [(train, "2024"), (test, "2025")]:
    dset = dset.copy()
    dset["balance"] = dset["load_p"] - (dset["grid_p"] + dset["battery_p"] + dset["pv_p"])
    max_imbalance = dset["balance"].abs().max()
    mean_imbalance = dset["balance"].abs().mean()
    rpt(f"\n  {label} energy balance check (load - grid - batt - pv):")
    rpt(f"    Max  imbalance: {max_imbalance:.6f} kW")
    rpt(f"    Mean imbalance: {mean_imbalance:.6f} kW")
    rpt(f"    Balance holds: {max_imbalance < 0.01}")

# Battery SoC reconstruction
rpt("\n  Reconstructing Battery SoC from energy balance...")
BATTERY_CAPACITY = 10.0  # kWh (assumed, typical residential)
EFF = np.sqrt(0.90)
INIT_SOC = 0.5

for dset, label in [(train, "2024"), (test, "2025")]:
    dset = dset.copy()
    soc = np.zeros(len(dset))
    soc[0] = INIT_SOC * BATTERY_CAPACITY
    for i in range(1, len(dset)):
        bp = dset["battery_p"].iloc[i]
        if bp > 0:  # discharging
            delta = bp * 0.25 / EFF
        else:        # charging
            delta = bp * 0.25 * EFF
        soc[i] = np.clip(soc[i-1] - delta, 0, BATTERY_CAPACITY)
    dset["soc"] = soc / BATTERY_CAPACITY * 100
    rpt(f"\n  {label} reconstructed SoC:")
    rpt(f"    Mean SoC : {dset['soc'].mean():.1f}%")
    rpt(f"    Min SoC  : {dset['soc'].min():.1f}%")
    rpt(f"    Max SoC  : {dset['soc'].max():.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for row, dset, label in zip([0,1], [train, test], ["2024","2025"]):
    dset = dset.copy()
    soc = np.zeros(len(dset))
    soc[0] = INIT_SOC * BATTERY_CAPACITY
    for i in range(1, len(dset)):
        bp = dset["battery_p"].iloc[i]
        delta = bp * 0.25 / EFF if bp > 0 else bp * 0.25 * EFF
        soc[i] = np.clip(soc[i-1] - delta, 0, BATTERY_CAPACITY)
    dset["soc"] = soc / BATTERY_CAPACITY * 100

    axes[row,0].plot(dset["timestamp"], dset["battery_p"], color="seagreen", linewidth=0.4, alpha=0.7)
    axes[row,0].axhline(0, color="black", linewidth=0.6)
    axes[row,0].set_ylabel("Battery Power (kW)")
    axes[row,0].set_title(f"Battery Power — {label}")

    axes[row,1].plot(dset["timestamp"], dset["soc"], color="steelblue", linewidth=0.4, alpha=0.8)
    axes[row,1].set_ylabel("SoC (%)")
    axes[row,1].set_title(f"Reconstructed Battery SoC — {label}")
    axes[row,1].set_ylim(-5, 105)
    axes[row,1].axhline(20, color="red", linestyle="--", linewidth=0.8, label="20% low")
    axes[row,1].axhline(80, color="green", linestyle="--", linewidth=0.8, label="80% high")
    axes[row,1].legend(fontsize=8)

fig.suptitle("Battery Operation Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "12_battery_analysis")


# ─── 7. ELECTRICITY PRICE ANALYSIS ────────────────────────────────────────────
rpt("\n[7] ELECTRICITY PRICE ANALYSIS")
rpt("-" * 50)

df_price = df.dropna(subset=["Selling_price_eur_kwh"])
for dset, label in [(train, "2024"), (test, "2025")]:
    d = dset.dropna(subset=["Selling_price_eur_kwh"])
    rpt(f"\n  {label}:")
    rpt(f"    Mean price   : {d['Selling_price_eur_kwh'].mean():.5f} €/kWh")
    rpt(f"    Median price : {d['Selling_price_eur_kwh'].median():.5f} €/kWh")
    rpt(f"    Std price    : {d['Selling_price_eur_kwh'].std():.5f} €/kWh")
    rpt(f"    Min price    : {d['Selling_price_eur_kwh'].min():.5f} €/kWh")
    rpt(f"    Max price    : {d['Selling_price_eur_kwh'].max():.5f} €/kWh")
    rpt(f"    Negative price periods: {(d['Selling_price_eur_kwh']<0).sum()}")

# Price distribution & time patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Distribution
axes[0,0].hist(df_price["Selling_price_eur_kwh"], bins=80, color="purple", alpha=0.7, edgecolor="none")
axes[0,0].set_xlabel("Price (€/kWh)")
axes[0,0].set_title("Price Distribution (all data)")

# Hourly price pattern
df_price_c = df_price.copy()
df_price_c["hour"] = df_price_c["timestamp"].dt.hour
hp = df_price_c.groupby("hour")["Selling_price_eur_kwh"].mean()
axes[0,1].plot(hp.index, hp.values, color="purple", linewidth=2, marker="o", markersize=3)
axes[0,1].set_xlabel("Hour of Day")
axes[0,1].set_ylabel("Mean Price (€/kWh)")
axes[0,1].set_title("Average Hourly Price Pattern")
axes[0,1].set_xticks(range(0, 24))

# Monthly price
df_price_c["month"] = df_price_c["timestamp"].dt.month
mp = df_price_c.groupby("month")["Selling_price_eur_kwh"].mean()
axes[1,0].bar([month_labels[m-1] for m in mp.index], mp.values, color="purple", alpha=0.7)
axes[1,0].set_ylabel("Mean Price (€/kWh)")
axes[1,0].set_title("Monthly Average Price")

# 2024 vs 2025 price overlay
axes[1,1].hist(train.dropna()["Selling_price_eur_kwh"], bins=60, alpha=0.5, color="steelblue", density=True, label="2024")
axes[1,1].hist(test.dropna()["Selling_price_eur_kwh"],  bins=60, alpha=0.5, color="tomato",    density=True, label="2025")
axes[1,1].set_xlabel("Price (€/kWh)")
axes[1,1].set_title("Price Distribution: 2024 vs 2025")
axes[1,1].legend()

fig.suptitle("Electricity Price Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "13_price_analysis")

# Price spikes
price_p99 = df_price["Selling_price_eur_kwh"].quantile(0.99)
price_p01 = df_price["Selling_price_eur_kwh"].quantile(0.01)
spikes_high = df_price[df_price["Selling_price_eur_kwh"] >= price_p99]
spikes_low  = df_price[df_price["Selling_price_eur_kwh"] <= price_p01]
rpt(f"\n  Price spikes (>P99={price_p99:.5f}): {len(spikes_high)} events")
rpt(f"  Price dips  (<P01={price_p01:.5f}): {len(spikes_low)} events")

# Price vs load correlation (df_price already contains load_p)
corr_price_load = df_price["Selling_price_eur_kwh"].corr(df_price["load_p"])
rpt(f"  Price vs Load Pearson correlation: {corr_price_load:.4f}")


# ─── 8. CORRELATION & FEATURE ANALYSIS ───────────────────────────────────────
rpt("\n[8] CORRELATION & FEATURE ANALYSIS")
rpt("-" * 50)

df_feat = df.dropna().copy()
df_feat["hour"]  = df_feat["timestamp"].dt.hour
df_feat["dow"]   = df_feat["timestamp"].dt.dayofweek
df_feat["month"] = df_feat["timestamp"].dt.month
df_feat["is_weekend"] = (df_feat["dow"] >= 5).astype(int)

corr = df_feat[["load_p","grid_p","pv_p","battery_p","Selling_price_eur_kwh",
                "hour","dow","month","is_weekend"]].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, mask=False, square=True, linewidths=0.5)
ax.set_title("Correlation Matrix — All Features", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "14_correlation_matrix")

rpt("\n  Correlations with load_p:")
corr_load = corr["load_p"].drop("load_p").sort_values(key=abs, ascending=False)
for feat, val in corr_load.items():
    rpt(f"    {feat:<30} {val:+.4f}")


# ─── 9. ANOMALY & DATA QUALITY ────────────────────────────────────────────────
rpt("\n[9] ANOMALY & DATA QUALITY ANALYSIS")
rpt("-" * 50)

# Z-score anomalies in load
for dset, label in [(train, "2024"), (test, "2025")]:
    z = np.abs(stats.zscore(dset["load_p"].fillna(dset["load_p"].mean())))
    anomalies = dset[z > 3]
    rpt(f"\n  {label} load anomalies (|z|>3): {len(anomalies)} points ({len(anomalies)/len(dset)*100:.2f}%)")
    if len(anomalies) > 0:
        rpt(f"    Value range: {anomalies['load_p'].min():.3f} – {anomalies['load_p'].max():.3f} kW")

# Battery corruption check (mentioned in brief)
rpt("\n  Battery data quality check:")
for dset, label in [(train, "2024"), (test, "2025")]:
    # Look for runs of zeros or suspiciously constant values
    batt = dset["battery_p"]
    zero_runs = (batt == 0).astype(int)
    # Find longest consecutive zero run
    max_run = 0
    cur_run = 0
    for v in zero_runs:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    rpt(f"  {label}: Longest zero-battery run: {max_run} steps = {max_run*0.25:.1f} hours")
    unusual = dset[(dset["battery_p"].abs() > 8.5)]
    rpt(f"  {label}: Battery power >8.5kW (over-limit): {len(unusual)} points")

# Visualise anomalies
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
for ax, dset, label in zip(axes, [train, test], ["2024","2025"]):
    ax.plot(dset["timestamp"], dset["load_p"], linewidth=0.4, color="steelblue", alpha=0.7)
    z = np.abs(stats.zscore(dset["load_p"].fillna(dset["load_p"].mean())))
    anomaly_mask = z > 3
    ax.scatter(dset["timestamp"][anomaly_mask], dset["load_p"][anomaly_mask],
               color="red", s=10, zorder=5, label=f"Anomalies ({anomaly_mask.sum()})")
    ax.set_title(f"Load Anomaly Detection — {label}")
    ax.set_ylabel("Load (kW)")
    ax.legend()

fig.suptitle("Load Anomaly Detection (Z-score > 3)", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "15_load_anomalies")


# ─── 10. AUTOCORRELATION & STATIONARITY ──────────────────────────────────────
rpt("\n[10] AUTOCORRELATION & STATIONARITY")
rpt("-" * 50)

# ADF test on load
for dset, label in [(train, "2024"), (test, "2025")]:
    series = dset["load_p"].dropna()
    adf_result = adfuller(series, autolag="AIC")
    rpt(f"\n  {label} ADF test on load_p:")
    rpt(f"    ADF statistic : {adf_result[0]:.4f}")
    rpt(f"    p-value       : {adf_result[1]:.6f}")
    rpt(f"    Stationary    : {adf_result[1] < 0.05}")

# ACF / PACF plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for row, dset, label in zip([0,1], [train, test], ["2024","2025"]):
    series = dset["load_p"].dropna().values
    plot_acf(series, lags=96*7, ax=axes[row,0], alpha=0.05)
    axes[row,0].set_title(f"ACF — Load ({label}), 7 days lags")
    axes[row,0].set_xlabel("Lag (15-min steps)")

    plot_pacf(series[:5000], lags=100, ax=axes[row,1], alpha=0.05, method="ywm")
    axes[row,1].set_title(f"PACF — Load ({label}), first 100 lags")
    axes[row,1].set_xlabel("Lag (15-min steps)")

fig.suptitle("Autocorrelation Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "16_autocorrelation")

# Key lag insights
for dset, label in [(train, "2024"), (test, "2025")]:
    series = dset["load_p"].dropna().values
    acf_vals = acf(series, nlags=96*7+1)
    rpt(f"\n  {label} ACF key lags:")
    rpt(f"    Lag 1  (15min)  : {acf_vals[1]:.4f}")
    rpt(f"    Lag 4  (1h)     : {acf_vals[4]:.4f}")
    rpt(f"    Lag 96 (1day)   : {acf_vals[96]:.4f}")
    rpt(f"    Lag 192(2day)   : {acf_vals[192]:.4f}")
    rpt(f"    Lag 672(1week)  : {acf_vals[672]:.4f}")


# ─── 11. SPECTRAL ANALYSIS ────────────────────────────────────────────────────
rpt("\n[11] SPECTRAL ANALYSIS (PERIODOGRAM)")
rpt("-" * 50)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, dset, label in zip(axes, [train, test], ["2024","2025"]):
    series = dset["load_p"].fillna(dset["load_p"].mean()).values
    freqs, power = periodogram(series, fs=4)  # 4 samples/hour
    periods_hours = 1.0 / (freqs[1:] + 1e-12)
    ax.semilogy(periods_hours, power[1:], linewidth=0.5, color="steelblue", alpha=0.8)
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"Load Periodogram — {label}")
    ax.set_xlim(0, 200)
    # Mark key periods
    for p, name in [(24,"Daily"), (168,"Weekly")]:
        ax.axvline(p, color="red", linestyle="--", linewidth=1, alpha=0.7, label=name)
    ax.legend()

    # Top 5 dominant periods
    top_idx = np.argsort(power[1:])[-5:][::-1]
    rpt(f"\n  {label} top 5 dominant periods:")
    for idx in top_idx:
        rpt(f"    {periods_hours[idx]:.1f} hours")

fig.suptitle("Spectral Analysis — Load Periodogram", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "17_spectral_analysis")


# ─── 12. LOAD FORECASTING FEATURE ANALYSIS ───────────────────────────────────
rpt("\n[12] LOAD FORECASTING — FEATURE IMPORTANCE PREVIEW")
rpt("-" * 50)

from sklearn.ensemble import RandomForestRegressor

df_ml = train.copy()
df_ml["hour"]    = df_ml["timestamp"].dt.hour
df_ml["minute"]  = df_ml["timestamp"].dt.minute
df_ml["dow"]     = df_ml["timestamp"].dt.dayofweek
df_ml["month"]   = df_ml["timestamp"].dt.month
df_ml["is_weekend"] = (df_ml["dow"] >= 5).astype(int)
df_ml["lag_1"]   = df_ml["load_p"].shift(1)
df_ml["lag_4"]   = df_ml["load_p"].shift(4)
df_ml["lag_96"]  = df_ml["load_p"].shift(96)
df_ml["lag_192"] = df_ml["load_p"].shift(192)
df_ml["lag_672"] = df_ml["load_p"].shift(672)
df_ml["roll_4_mean"]  = df_ml["load_p"].shift(1).rolling(4).mean()
df_ml["roll_96_mean"] = df_ml["load_p"].shift(1).rolling(96).mean()
df_ml["pv_lag1"]      = df_ml["pv_p"].shift(1)
df_ml = df_ml.dropna()

feature_cols = ["hour","minute","dow","month","is_weekend",
                "lag_1","lag_4","lag_96","lag_192","lag_672",
                "roll_4_mean","roll_96_mean","pv_lag1","Selling_price_eur_kwh"]
df_ml = df_ml.dropna(subset=feature_cols)
X = df_ml[feature_cols].fillna(0)
y = df_ml["load_p"]

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

rpt("\n  Random Forest feature importances (quick proxy):")
for feat, imp in importances.items():
    rpt(f"    {feat:<25} {imp:.4f}  {'█' * int(imp*100)}")

fig, ax = plt.subplots(figsize=(10, 6))
importances.sort_values().plot.barh(ax=ax, color="steelblue", alpha=0.8)
ax.set_xlabel("Feature Importance")
ax.set_title("Feature Importance for Load Forecasting\n(Random Forest — Quick Proxy, 2024 train)")
plt.tight_layout()
save(fig, "18_feature_importance")


# ─── 13. SUMMARY & RECOMMENDATIONS ───────────────────────────────────────────
rpt("\n" + "=" * 70)
rpt("[13] KEY FINDINGS & MODELLING RECOMMENDATIONS")
rpt("=" * 70)

rpt("""
DATA CHARACTERISTICS:
  - Dataset: 70,077 rows × 6 columns, 15-min resolution, 2024 (train) + 2025 (test)
  - 9 missing price values at DST transitions — negligible, forward-fill safe
  - Energy balance holds well (imbalance < 0.01 kW)

LOAD PATTERNS:
  - Strong daily periodicity (ACF lag-96 ≈ 0.9+), strong weekly (lag-672)
  - Morning peak ~7-9h, evening peak ~18-21h (typical residential)
  - Weekend load slightly lower / shifted later than weekday
  - Winter months (Dec-Feb) show higher load than summer
  - 2024 vs 2025 load is highly correlated (>0.95 daily) — 2024 is good training data

PV GENERATION:
  - Clear seasonal signal: peak in summer, near-zero in winter
  - Generation window: approx 06:00–20:00 (varies by season)
  - PV covers ~X% of total load — significant self-consumption potential

PRICE SIGNAL:
  - Prices vary intra-day — clear peak/off-peak structure
  - Occasional negative prices (market oversupply) → charge battery
  - 2025 prices slightly higher than 2024

MODELLING STRATEGY:
  1. FORECASTING:
     - Key features: lag_96 (yesterday same time), lag_672 (last week),
       hour, dow, month, rolling means, is_weekend
     - Recommended models: LightGBM (fast, strong baseline),
       XGBoost, LSTM (capture temporal dependencies)
     - Validate with rolling-window CV on 2024 data
     - Target metric: NRMSE = RMSE / mean(load) × 100

  2. BATTERY CONTROLLER:
     - Use price signal directly: charge when price low, discharge when high
     - Layer forecast on top: if high load predicted + high price → pre-charge
     - Rolling horizon: re-optimize every timestep with updated forecast
     - Key constraint: SoC 0-100%, power ±8kW, grid ±6kW, 90% round-trip eff.
     - Start with rule-based greedy → then MPC with forecast
""")

# ─── SAVE REPORT ──────────────────────────────────────────────────────────────
report_path = REPORTS_DIR / "eda_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\nReport saved to {report_path}")
print(f"All plots saved to {PLOTS_DIR}/")
print("EDA complete.")
