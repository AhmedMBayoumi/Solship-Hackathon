# Solship Energy AI Hackathon 2026 — Submission

**Team**: AhmedBayoumi
**Site**: Sondrio, Italy (residential, 9 kWp PV, 16 kWh / ±8 kW battery, ±6 kW grid)
**Test window**: April 2025 + September 2025

---

## 🏆 Headline Results

| Metric                                                    | Value          |
|-----------------------------------------------------------|----------------|
| **Forecast NRMSE (Apr+Sep 2025)**                         | **61.46 %**    |
| Forecast RMSE                                             | 0.5538 kW      |
| Forecast MAE                                              | 0.3303 kW      |
| Forecast MAPE                                             | 53.73 %        |
| Forecast R²                                               | 0.714          |
| **Net bill our MPC (Apr+Sep 2025)**                       | **EUR -18.79** |
| Baseline A bill (existing controller)                     | EUR -7.44      |
| **Savings vs Baseline A**                                 | **+11.35 EUR (+152.6 %)** |
| Oracle bill (perfect foresight, H=96)                     | EUR -20.15     |
| Oracle gap                                                | 1.36 EUR       |

We are within EUR 1.36 of the theoretical maximum.

---

## 📂 Where everything is

```
submission/
├── README.md                                  ← this file
├── data/
│   ├── Solship_Hackathon_Submission.xlsx      ← OFFICIAL SUBMISSION (with metrics + plots embedded)
│   ├── forecast_april_2025.csv                ← per-step April forecast vs actual
│   ├── forecast_september_2025.csv            ← per-step September forecast vs actual
│   ├── bills_and_savings.csv                  ← controller bills + savings vs A
│   └── master_metrics.csv                     ← all metrics in one file
├── plots/
│   ├── forecast_april_2025.png                ← April load vs forecast time series + residual
│   ├── forecast_september_2025.png            ← September load vs forecast time series + residual
│   ├── error_hist_april_2025.png              ← April error distribution histograms
│   ├── error_hist_september_2025.png          ← September error distribution histograms
│   ├── scatter_april_2025.png                 ← April predicted vs actual scatter
│   ├── scatter_september_2025.png             ← September predicted vs actual scatter
│   ├── bills_comparison.png                   ← controller bills bar chart
│   ├── results_horizon_sensitivity.png        ← Extension 1: 10-horizon sweep
│   ├── dispatch_march_week3.png               ← mandatory March Week 3 dispatch (5-panel)
│   ├── forecast_pred_vs_actual.png            ← global pred-vs-actual scatter
│   ├── forecast_error_by_hour.png             ← error breakdown by hour-of-day
│   ├── forecast_feature_importance.png        ← top-30 LightGBM features
│   ├── forecast_spike_analysis.png            ← spike detection / under-prediction analysis
│   ├── eda_load_heatmap.png                   ← hour×dow load heatmap
│   ├── eda_price_analysis.png                 ← Italian TOU tariff analysis
│   └── eda_autocorrelation.png                ← load ACF / PACF
└── code/
    ├── src/                                   ← source code (forecaster + MPC + LP)
    ├── scripts/                               ← training + evaluation scripts
    ├── modal_scripts/                         ← cloud GPU training (Tesla T4)
    └── models_pretrained/                     ← saved model artefacts (JSON)
```

---

## 🧠 Model architecture (best result)

**Forecaster**: bagging of 12 LightGBM regressors trained walk-forward.

| Setting                   | Value                                    |
|---------------------------|------------------------------------------|
| Loss                      | Huber (alpha=0.9)                        |
| Trees                     | 3000 per bag                             |
| Regularization            | heavy: num_leaves 7-63, max_depth 3-6, reg_alpha/lambda 0.5-5, subsample 0.6-0.9 |
| Train data (April model)  | 2024 (all) + 2025 Jan-Feb (val: 2025 Mar)|
| Train data (Sept model)   | 2024 (all) + 2025 Jan-Aug (val: 2025 last half of Aug) |
| Features                  | 92 (lags 1..2016, recent-change deltas, rolling 4/8/16/96/384/672 stats, net-load lags, weather + HDD/CDD, Fourier 24h/12h/8h/4h/annual, calendar, tariff_band, holidays) |
| Inference                 | Average of 12 bagged predictors          |

**Optimizer**: rolling-horizon MPC + LP relaxation.

| Setting     | Value                                                      |
|-------------|------------------------------------------------------------|
| Horizon     | H = 96 (1 day = battery cycle, per supervisor)             |
| Solver      | scipy.optimize.linprog (HiGHS) — replaces CVXPY (Win crash)|
| LP vars     | 5H + 1: charge/discharge/import/export per step + soc[0..H]|
| Causal      | Per hackathon rules: actual load observed at current step  |
| Constraints | Power balance, SoC dynamics with eff=√0.9, |p_bat|≤8, |p_g|≤6, soc∈[0,1] |
| Wall time   | ~52 s for both Apr+Sep (full month rolling)                |

---

## 🔑 Two critical bug fixes that drove the bill from EUR +9 to EUR -19

1. **Grid clamp sign bug** in `mpc_loop.py`: when |p_grid| > 6, we were doing
   `p_bat -= delta` which made the violation worse and silently broke the
   energy balance. Fixed to `p_bat += delta`.
2. **Forecast at k=0**: per the rules, current load is OBSERVED.
   Was using forecast for the current step. Now the LP horizon's first
   element is the actual load. This single fix moved bills from
   EUR +8.38 to EUR -19.16 at H=96.

---

## 📊 What we tried (full experimental log)

11 model variants converged to ~62 % NRMSE — see `code/scripts/` and the
`outputs/reports/training_log.txt` in the parent project for the complete
experimental log. Highlights:

| Approach                                   | Test NRMSE |
|--------------------------------------------|-----------:|
| Persistence lag-1                          |     72.30 %|
| LightGBM v1 (39 features)                  |     64.23 %|
| LightGBM v2 + heavy reg + huber            |     62.08 %|
| Bagging 12× LGBM (v2 features)             |     61.97 %|
| ML+DL gated fusion (LGBM/XGB/CAT/MLP)      |     62.17 %|
| ML+DL gated fusion v2 (spike-aware)        |     62.27 %|
| ML+DL gated fusion v3 (specialists)        |     62.31 %|
| **Bagging walkforward (THIS SUBMISSION)**  | **61.46 %**|

Why every approach plateaus at ~62 % for single-residential 15-min:
- CV = std/mean = 1.15 — std exceeds mean
- Lag-1 ACF = 0.80 → pure-from-lag1 floor is ~69 % NRMSE
- Residual ACF after our model is within ±0.1 — we capture all detectable
  temporal structure; remaining variance is appliance-switching noise
  that is irreducibly random without sub-meter / occupancy data.
- Industry SOTA (TFT, N-BEATS, DeepAR) hits the same ceiling on single
  homes; they only break it via multi-house aggregation or transfer
  learning from millions of meters.

---

## ▶️ How to reproduce

```bash
# 1. Install
pip install -r requirements.txt

# 2. Build features (creates data/features/features_v2_*.parquet)
python -m src.features.build_features_v2

# 3. Train walkforward bagging (April + September models, ~5 min)
python scripts/train_walkforward_bagging.py

# 4. Run rolling-horizon MPC at H=96 (~1 min)
python scripts/run_mpc_walkforward.py

# 5. Build submission spreadsheet + plots
python scripts/build_submission_xlsx.py
```

---

## 📐 Battery system constraints (verified)

- Battery: 16 kWh usable, ±8 kW max charge/discharge, round-trip eff 90 %
- Grid: ±6 kW max import/export
- SoC reset: 50 % at start of April 1 and September 1 (independent windows)
- Verified energy balance in saved dispatches: max |error| = 0.0000 kW
- Verified bound violations: 0 timesteps over |p_bat|≤8, |p_g|≤6, soc∈[0,1]

---

## 🌐 Open-Meteo weather

Sondrio (lat 46.17, lon 9.87) historical weather pulled at 15-min
resolution from `https://archive-api.open-meteo.com/v1/archive`.
Variables: `temperature_2m`, `shortwave_radiation`, `cloud_cover`,
`relative_humidity_2m`, plus computed HDD = max(18-T, 0) and
CDD = max(T-24, 0).

---

*Submission ready 2026-05-09.*
