# Energy AI Hackathon 2026 — 🥇 Winning Submission

**Team 25 · Ahmed Mohamed Bayoumi**
*Zewail City of Science and Technology · Powered by Solship*

Forecast-aware Model Predictive Control for residential battery dispatch on
an Italian site in Sondrio (9 kWp PV, 16 kWh / 8 kW battery, 6 kW grid).

---

## Headline results

| Metric | Value |
|---|---|
| **Forecasting NRMSE — 2025 test (Apr + Sep)** | **52.17 %** |
| **Forecasting NRMSE — surprise dataset (Mar 2026)** | **32.16 %** |
| **MPC bill (H = 96)** | **−€18.89** |
| Baseline A (historical controller) | −€7.44 |
| Oracle (perfect foresight) | −€20.14 |
| **Savings vs Baseline A** | **+€11.45 (+154 %)** |
| **Oracle gap captured** | **90 %** |

---

## What the project does

Two coupled problems, solved end-to-end:

1. **Load forecasting** — predict household electrical load at 15-min resolution
   over the next 24 hours, using only data available at decision time
   (no leakage).
2. **Battery dispatch** — at every 15-min step, decide how much to charge or
   discharge to minimise the electricity bill under the Italian three-band
   tariff (F1 / F2 / F3), respecting hard physics limits.

### Forecasting pipeline

```
INPUT
  ├─ raw signals: load · pv · weather · ARPA Lombardia local radiation
  └─ Italian calendar (F1/F2/F3 tariffs, holidays, bridge days)
        ↓
v7 FEATURES   (153 columns)
   lag · rolling · wavelet decomposition · clear-sky physics · cyclical
        ↓                          ↓
LSTM AUTOENCODER             8-BAG LIGHTGBM
(PyTorch on Modal T4)         (online retraining + walkforward,
4 → 32 → 16-dim bottleneck     light-reg configs, Huber loss)
        ↓                          ↓
        └────────────┬─────────────┘
                     ↓
            5-FOLD CV-NNLS BLEND
        (honest stacking — per-fold weights
         from out-of-fold residuals)
                     ↓
            MA(3) + α-RESCALE SMOOTHING
        (variance-preserving, no test actuals used)
                     ↓
                FINAL FORECAST
```

### Dispatch pipeline (MPC)

```
At every 15-min step:
  ① Get H = 96-step forecast (24 h look-ahead)
  ② Solve LP via scipy.optimize.linprog (HiGHS)
        Variables: p_chg, p_dis, p_imp, p_exp, soc   (5H + 1 = 481)
        Objective: min Σ [buy·p_imp − sell·p_exp] · Δt
        Constraints: SoC ∈ [0,1] · ±8 kW battery · ±6 kW grid ·
                     energy balance · SoC dynamics
  ③ Execute first action only, discard the rest of the plan
  ④ Update SoC from realised dispatch, advance to t+1
```

---

## Folder layout

```
.
├── README.md                              ← you're reading it
│
├── data/
│   ├── raw/                                Hackathon dataset (1st site)
│   ├── processed/                          Cleaned & renamed (load_kw, pv_kw, …)
│   ├── external/                           ARPA Lombardia radiation pulls, clearsky
│   └── features/                           v7 / surprise feature parquets + cols files
│
├── src/                                    Core library code
│   ├── controller/                         MPC loop + LP optimizer (scipy HiGHS)
│   ├── eval/                               Bill + metric utilities
│   ├── features/                           v6 / v7 / clearsky / surprise feature builders
│   └── data/                               Raw → processed pipeline
│
├── scripts/                                Top-level runnable scripts
│   │
│   │ Forecasting
│   ├── online_retraining_lgbm.py             8-bag LGBM, retrain every 3 days (60.65%)
│   ├── local_v10_dl_extractor.py             LSTM-AE + 8-bag LGBM (CPU)
│   ├── local_v11_dl_v2.py                    upgraded LSTM-AE (forecast-aux loss)
│   ├── final_blend.py                        5-fold CV-NNLS blend  →  52.17% NRMSE
│   │
│   │ MPC controller
│   ├── run_mpc_blend.py                      Rolling-horizon MPC with blend forecast
│   ├── run_mpc_final.py / run_mpc_rolling.py / run_mpc_walkforward.py
│   ├── compute_baselines_and_plots.py        Baseline A/B + Oracle + 4-line bill chart
│   ├── march_week3_dispatch.py               Mandatory March W3 dispatch plot
│   │
│   │ External data + diagnostics
│   ├── fetch_arpa_radiation.py               Pull ARPA Lombardia sensor 2098 (Sondrio)
│   ├── plot_final_forecast.py                Apr/Sep forecast vs actual
│   ├── full_diagnostic_report.py             Comprehensive forecasting diagnostics
│   ├── verify_data.py · analyze_*.py · diagnose_forecast_quality.py
│   │
│   │ Day-2 generalisation
│   ├── surprise_lstm_ae_local.py             LSTM-AE on the surprise (2nd) site
│   ├── surprise_postprocess_and_report.py    Surprise blend + smoothing → 32.16% NRMSE
│   ├── online_retraining_surprise.py         Online retraining on surprise data
│   ├── predict_week_ahead_v3.py              1-week-ahead direct multi-step model
│   │
│   │ Deliverables (Excel / PPTX / PDF)
│   ├── build_apr_sep_submission.py           April+Sept Excel (5 sheets)
│   ├── build_surprise_submission.py          Surprise dataset Excel
│   ├── build_weekahead_excel.py              Week-ahead Excel
│   ├── draw_architecture.py                  Forecasting model infographic
│   ├── build_presentation.py                 6-slide PPTX builder
│   └── build_presentation_pdf.py             PDF version of the deck
│
├── modal_scripts/                          Modal-GPU training (LSTM-AE on T4)
│   ├── train_v10_lstm_ae.py                  Original-site LSTM-AE on Modal T4
│   ├── train_surprise_lstm_ae.py             Surprise-site LSTM-AE on Modal T4
│   └── train_fusion_walkforward.py
│
├── outputs/                                Forecasts, bills, plots, models
│   ├── forecasts/                            CSVs of every predictor we kept
│   ├── plots/                                forecast / surprise / presentation plots
│   ├── models/                               metric JSONs
│   ├── mpc_blend_H96.parquet                 Final dispatch trajectory (our submission)
│   ├── mpc_oracle_H96.parquet                Perfect-foresight oracle trajectory
│   ├── horizon_sweep_*.csv                   H ∈ {4,8,16,24,48,96} sensitivity
│   ├── reports/                              Gemini Q&A, diagnostics
│   └── Solship_Hackathon_Presentation.pptx
│
├── day 2/                                  All Day-2 final deliverables
│   ├── Solship_Hackathon_Presentation.pptx       6-slide deck (editable)
│   ├── Team 25- Ahmed Mohamed Bayoumi.pdf         6-slide deck (PDF)
│   ├── submission 1/                              Week-ahead forecast (Jan 1-7, 2026)
│   ├── submission 2/                              Surprise dataset forecast (Mar 2026)
│   └── submission_apr_sep/                        Original April + September 2025
│       ├── Apr_Sep_2025_Forecast.xlsx              5 sheets: Team · Forecast & Dispatch
│       │                                           · Metrics · Savings · Plots
│       ├── Apr_Sep_2025_Forecast.csv
│       ├── Apr_Sep_2025_Forecast_smoothed.png
│       └── Apr_Sep_2025_Dispatch.png
│
├── submission/                             Earlier (Day-1) submission package
├── notebooks/                              Initial exploratory work
└── resources/                              PDFs (problem statement, presentation brief)
```

---

## How to reproduce the headline result

```bash
# 1.  Build the v7 feature set (one-off, slow)
python src/features/build_features_v7.py

# 2.  Train the LSTM-AE feature extractor (Modal T4 GPU recommended; ~3s on T4)
modal run modal_scripts/train_v10_lstm_ae.py

# 3.  Train the 8-bag LGBM with online retraining locally
python scripts/online_retraining_lgbm.py

# 4.  Blend the two via 5-fold CV-NNLS + apply smoothing
python scripts/final_blend.py

# 5.  Run the rolling-horizon MPC at H = 96
python scripts/run_mpc_blend.py

# 6.  Compute baselines + Oracle + the 4-line cumulative bill chart
python scripts/compute_baselines_and_plots.py

# 7.  Mandatory March Week 3 dispatch plot
python scripts/march_week3_dispatch.py

# 8.  Build the final submission deliverables
python scripts/build_apr_sep_submission.py
python scripts/build_surprise_submission.py
python scripts/build_weekahead_excel.py

# 9.  Build the presentation
python scripts/draw_architecture.py
python scripts/build_presentation.py
python scripts/build_presentation_pdf.py
```

---

## Why each design choice

| Choice | Rationale |
|---|---|
| **8-bag LightGBM** | Diverse light-reg configs cancel noise; far cheaper than a single deep model |
| **LSTM autoencoder feature extractor** | Bottleneck captures multi-channel regime info that tabular lag features miss; errors decorrelate with LGBM → useful in the blend |
| **5-fold CV-NNLS blend** | Honest stacking — per-fold weights from out-of-fold residuals, no in-sample fit |
| **MA(3) symmetric + α=1.05 rescale** | Smoothing exploits ≈white-noise prediction residuals (lag-1 ACF +0.02); α restores variance using prediction stats only (no test actuals) |
| **Coring at threshold 0.4 kW** | Pareto fix — smooth where \|raw − MA(3)\| < 0.4 (= noise), keep raw where larger (= real spike). Improves NRMSE AND bill simultaneously |
| **Rolling-horizon MPC, H = 96** | 24h window captures the full F3-night → F1-day arbitrage cycle; diminishing returns past 96 |
| **Linear program (no MIP)** | Convex, polynomial-time, single global optimum; at non-negative tariffs LP picks one direction naturally |
| **scipy.optimize.linprog (HiGHS)** | Open-source, deterministic, ~10 ms per H=96 solve; stable on Windows |

---

## Acknowledgements

- **Solship** for the well-designed problem and dataset.
- **Zewail City of Science and Technology** for hosting the hackathon.
- **ARPA Lombardia** for the open Sondrio radiation sensor data
  (dataset `cxym-eps2`, station 2098 — Sondrio Fond. Fojanini).
- Open-source toolchain: LightGBM, PyTorch, scipy HiGHS, pandas, NumPy,
  PyWavelets, openpyxl, matplotlib, python-pptx.

🥇 *1st place · Energy AI Hackathon 2026*
