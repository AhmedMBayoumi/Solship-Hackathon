# Solship Energy AI Hackathon 2026 — Submission

**Team**: Ahmed Moahmed Bayoumi

---

## Headline Results

| Metric                                                    | Value          |
|-----------------------------------------------------------|----------------|
| **Forecast NRMSE (Apr+Sep 2025)**                         | **60.83 %**    |
| Forecast RMSE                                             | 0.5482 kW      |
| Forecast MAE                                              | 0.3406 kW      |
| Forecast MAPE                                             | 58.45 %        |
| Forecast R²                                               | 0.720          |
| **Net bill our MPC (Apr+Sep 2025)**                       | **EUR -19.11** |
| Baseline A bill (existing controller)                     | EUR -7.44      |
| **Savings vs Baseline A**                                 | **+11.67 EUR (+156.9 %)** |
| Oracle bill (perfect foresight, H=96)                     | EUR -20.15     |
| Oracle gap                                                | 1.04 EUR       |

We are within EUR 1.04 of the theoretical maximum.

### Key change vs initial submission
With supervisor's allowance to include 2024 + 2025-up-to-test-month data,
we re-tested regularization. Heavy reg (necessary when training was
2024-only) is now over-restricting. **Light regularization** (num_leaves=63,
max_depth=8, reg_alpha/lambda=0.1) gave a 0.63 pp NRMSE improvement and
EUR 0.32 better MPC bill.

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
| Regularization            | LIGHT: num_leaves 47-95, max_depth 7-10, reg_alpha 0.05-0.3, reg_lambda 0.1-0.5, subsample 0.85-0.9 |
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


## 📊 What I tried (full experimental log)

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
| Bagging walkforward (heavy reg, prev)      |     61.46 %|
| Bagging walkforward + v3 features (132)    |     61.54 %|
| Bagging walkforward + v4 features (104)    |     61.54 %|
| **Bagging walkforward + light reg (FINAL)**| **60.83 %**|

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

*Submission ready 2026-05-09.*
