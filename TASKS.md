# SOLSHIP HACKATHON 2026 — MASTER TASK LIST

**Time budget:** 2 days × 6 hours = 12 hours total
**Submission deadline:** Day 2 at 15:00 (slides + code)
**Presentation:** 3 minutes, max 6 slides

---

## CRITICAL NOTES UP FRONT

### 1. We are testing MORE horizons than the PDF suggests
The participant brief (Extension 1) requires **at least 3** horizons, suggesting `H ∈ {4, 24, 96}` (1h / 6h / 24h).
**We will run 10 horizons** to find the actual diminishing-returns knee precisely:

```
H = 1   (15 min — myopic, lower bound)
H = 4   (1 hour, PDF example)
H = 8   (2 hours)
H = 16  (4 hours)
H = 24  (6 hours, PDF example)
H = 48  (12 hours)
H = 96  (24 hours, PDF example — covers full diurnal cycle)
H = 192 (48 hours)
H = 288 (72 hours)
H = 672 (1 week — captures weekday/weekend transitions)
```

Why: bonus only requires 3, but doing 10 lets us actually *find* the optimum and gives us a stronger justification on slide 2 ("we tested 10 horizons; H=X is the knee"). At H=672 the LP gets large but with the LP relaxation it should still solve in seconds.

### 2. LP RELAXATION (single biggest speedup)
Drop the binary `is_charging` / `is_discharging` variables in the optimizer.
Reason: round-trip efficiency < 100% means simultaneous charge+discharge is always sub-optimal — the LP solver naturally picks one direction. **MPC becomes 50–100× faster**, which makes the 10-horizon sweep tractable.

### 3. HARD RULES (auto-disqualify if broken)
- NO 2025 data in training / tuning / validation. Period.
- NO batch optimizer over full 2025 — must be causal rolling-horizon.
- Must reconstruct SoC from energy balance (raw `p_battery_kw` is corrupted in 2025).
- Must produce March Week 3 2025 dispatch plot.
- Must submit by 15:00 Day 2.

---

## PHASE 0 — DONE (do not redo)

- [x] Modal.com GPU verified (Tesla T4, $30 credit confirmed)
- [x] Project structure + private GitHub repo (`Solship-Hackathon`)
- [x] Deep EDA with 18 plots + report (`outputs/reports/eda_report.html`)
- [x] Both PDFs parsed; key facts in `resources/REFERENCE.txt`
- [x] Processed dataset built: `data/processed/dataset_processed.csv`
  (10 cols: timestamp, load_kw, pv_kw, p_battery_kw, grid_kw, buy_price, sell_price, tariff_band, is_holiday, net_load)
- [x] Baseline A (EUR -7.57) and Baseline B (EUR +68.74) computed for Apr+Sep 2025
- [x] Training log scaffold: `outputs/reports/training_log.txt`
- [x] Loader module: `src/data/loader.py` (load_train, load_test, load_split)
- [x] Gemini's external strategy doc reviewed and reconciled

---

## PHASE 1 — FEATURE ENGINEERING (Day 1, ~1.5h)

### 1.1 Fetch Sondrio weather from Open-Meteo  [HIGH]
- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Lat 46.17, Lon 9.87, timezone Europe/Rome
- Variables: `temperature_2m`, `shortwave_radiation`, `cloud_cover`, `relative_humidity_2m`
- Date range: 2024-01-01 → 2025-12-31 (inclusive)
- Resolution: hourly → linear-interpolate to 15-min to match dataset
- Output: `data/external/sondrio_weather.csv`

### 1.2 Compute HDD/CDD features  [MED]
- HDD = max(18 − T, 0)  (heating threshold 18°C, Italian residential standard)
- CDD = max(T − 24, 0)  (cooling threshold 24°C)
- Add columns to feature matrix. Better than raw temp for thermostatic load.

### 1.3 Extend holiday list  [LOW]
- Add June 19 (Sondrio patron saint, Saints Gervasius and Protasius) — local
- Add "ponte" (bridge) flag: 1 if holiday is Tue or Thu and adjacent Mon/Fri
- Verify Apr 21 2025 (Easter Monday — already F3) is correctly flagged

### 1.4 Build full feature matrix  [HIGH]
- Module: `src/features/build_features.py`
- Output: `data/features/features_2024.parquet` and `features_2025.parquet`
- Columns:
  - **Lags** (load): lag_1, lag_4, lag_8, lag_96, lag_192, lag_672
  - **Lags** (pv): pv_lag_1, pv_lag_96
  - **Rolling means** (load): roll_4_mean, roll_16_mean, roll_96_mean
  - **Calendar**: hour, dow, month, day_of_year, week_of_year, is_weekend
  - **Holiday**: is_holiday, is_bridge, is_sondrio_local
  - **Tariff**: tariff_band (label encoded F1=0, F2=1, F3=2), buy_price
  - **Fourier**: sin/cos of hour (24h cycle, 12h, 8h), sin/cos of day-of-year
  - **Weather**: temperature_2m, shortwave_radiation, cloud_cover, relative_humidity, hdd, cdd
- DO NOT include: `p_battery_kw`, `grid_kw`, `sell_price` (controller doesn't have sell_price for forecasting; only for optimization)
- Compute lags using shift() on full timeline so 2025 lags reach into late-2024 context window. This is causal and legal.

### 1.5 Verify no train/test leakage  [HIGH]
- Assert: no 2025 rows used to compute any normalization stat
- Assert: feature pipeline is reproducible (set random seeds)

---

## PHASE 2 — LOAD FORECASTING (Day 1, ~2.5h)

### 2.1 Validation split  [HIGH]
- Train fold: 2024 except Apr + Sep (10 months)
- Validation: 2024 April + 2024 September (mirrors test months exactly)
- Test (final report only): 2025 April + 2025 September
- Surprise (Day 2 13:00): unseen second site, run model with no retrain

### 2.2 LightGBM baseline  [HIGH]
- Module: `src/models/lgbm_model.py`
- Hyperparams: n_estimators=2000, lr=0.05, num_leaves=63, max_depth=8, early_stopping=100
- Run on Modal (T4 GPU not strictly needed; LightGBM is CPU-fast — but Modal CPU is fine)
- Log to `training_log.txt` as [EXP-001]
- Target NRMSE: < 50% on validation

### 2.3 XGBoost baseline  [HIGH]
- Module: `src/models/xgb_model.py`
- Same train/val split, similar hyperparams
- Log to `training_log.txt` as [EXP-002]

### 2.4 CatBoost baseline  [MED]
- Module: `src/models/cat_model.py`
- Native handling of categorical features (tariff_band, hour, dow, month)
- Log to `training_log.txt` as [EXP-003]

### 2.5 Hyperparameter tuning  [MED]
- Use Optuna: 50 trials per model on Modal
- Tune per-model: lr, num_leaves/depth, reg_alpha, reg_lambda, subsample, colsample
- Time budget: 30 min on T4 GPU
- Log best NRMSE per model

### 2.6 Ensemble  [HIGH]
- Simple weighted average: w_lgb × LGB + w_xgb × XGB + w_cat × CAT
- Optimize weights on validation set
- Log final ensemble NRMSE on Apr 2024 + Sep 2024
- This is our "production" forecast model

### 2.7 Optional: LSTM if time allows  [LOW]
- Only if Phase 2.1–2.6 finished early
- Sequence input: last 96 timesteps (24h)
- Output: next 96 timesteps
- Modal T4 makes this fast

### 2.8 Forecast generation  [HIGH]
- Run final ensemble on Apr + Sep 2025 features
- Save predictions: `data/forecasts/load_forecast_2025.parquet`
- Compute final RMSE / MAE / NRMSE on Apr+Sep 2025

---

## PHASE 3 — BATTERY OPTIMIZER (Day 1 end + Day 2 start, ~2.5h)

### 3.1 LP formulation in CVXPY  [HIGH]
- Module: `src/controller/lp_optimizer.py`
- **Decision variables** (per timestep t in horizon):
  - `p_batt_charge[t]`  ≥ 0  (kW, charging power, positive)
  - `p_batt_disch[t]`   ≥ 0  (kW, discharging power, positive)
  - `p_grid_imp[t]`     ≥ 0  (kW, grid import)
  - `p_grid_exp[t]`     ≥ 0  (kW, grid export)
  - `soc[t]`            in [0, 1]
- **Constraints**:
  - Power balance: load[t] + p_batt_charge[t] + p_grid_exp[t] = pv[t] + p_batt_disch[t] + p_grid_imp[t]
  - SoC dynamics: soc[t+1] = soc[t] + (p_batt_charge[t] × √0.9 − p_batt_disch[t] / √0.9) × 0.25 / 16.0
  - p_batt_charge[t] ≤ 8.0
  - p_batt_disch[t] ≤ 8.0
  - p_grid_imp[t] ≤ 6.0
  - p_grid_exp[t] ≤ 6.0
- **Objective**:
  - minimize Σ [ p_grid_imp[t] × buy_price[t] − p_grid_exp[t] × sell_price[t] ] × 0.25
- **Why no binaries**: Round-trip eff = 0.9 < 1.0 → simultaneous charge+discharge always wastes energy → LP solver naturally picks one direction. **This is the key speedup.**
- Solver: HiGHS (free, fast) via CVXPY

### 3.2 Perfect-foresight oracle test  [HIGH]
- Run LP once over full April 2025 with H=2880 (whole month) using ACTUAL load
- Run again over full September 2025 with H=2880
- This gives the **upper bound** of achievable bill (no forecast error)
- Save oracle_bill_apr, oracle_bill_sep
- Sanity check: oracle bill should be more negative than Baseline A

### 3.3 Rolling-horizon MPC loop  [HIGH]
- Module: `src/controller/mpc_loop.py`
- Pseudocode:
  ```
  soc = 0.5  # 8 kWh / 16 kWh
  for t in range(start, end):
      load_forecast = forecast_model.predict(t, H)
      pv_actual_window = pv[t : t+H]            # PV is "known" forward (perfect)
      buy_window = buy_price[t : t+H]
      sell_window = sell_price[t : t+H]
      sol = lp_solve(load_forecast, pv_actual_window, buy_window, sell_window, soc)
      execute = sol.p_batt_disch[0] - sol.p_batt_charge[0]
      soc = update_soc(soc, execute)
      log(t, execute, grid_actual)
  ```
- Note: PDF uses load forecast but doesn't say PV is forecast — assume PV actuals are known. Confirm with mentor at Day 1 end. (If PV must also be forecast, build a tiny PV regression on 2024 sun/cloud features.)
- Resets at start of each test month: SoC = 50% on Apr 1 00:00 and Sep 1 00:00 (independent)

### 3.4 Single-run with chosen H (e.g., H=24)  [HIGH]
- Run MPC over April 2025 → bill_apr
- Run MPC over September 2025 → bill_sep
- Total bill = bill_apr + bill_sep
- vs Baseline A (-7.57): savings in € and %
- Log to `training_log.txt` as [OPT-001]

---

## PHASE 4 — HORIZON SENSITIVITY (Extension 1, +5 pts) (Day 2, ~1.5h)

### 4.1 Sweep 10 horizons  [HIGH]
- For H in [1, 4, 8, 16, 24, 48, 96, 192, 288, 672]:
  - Run MPC on April 2025 → bill_apr_H
  - Run MPC on September 2025 → bill_sep_H
  - Record total bill, savings vs Baseline A, wall-clock time
- Output: `outputs/horizon_sweep.csv`

### 4.2 Plot horizon vs bill curve  [HIGH]
- X-axis: H (log scale)
- Y-axis (left): total bill (€)
- Y-axis (right): wall time (s)
- Mark the "knee" where additional H gives < 1% savings improvement
- Save: `outputs/plots/19_horizon_sensitivity.png`

### 4.3 Pick final H + justify  [HIGH]
- Choose H at the knee (likely H=24 or H=96 based on intuition)
- Document justification for slide 2

---

## PHASE 5 — GENERALIZATION GUARDS (Day 1 throughout, ~0.5h)

- All forecast features must work on raw `(timestamp, load_kw, pv_kw, buy_price, sell_price)` only
- No site-specific tuning beyond what's encoded in 2024 training
- Holiday calendar generalizes to any Italian site (Sondrio patron saint can be safely included or zero'd)
- Test the pipeline on a held-out 2024 chunk treated as "another site" to make sure it runs end-to-end with no errors

---

## PHASE 6 — DELIVERABLES (Day 2, ~1.5h)

### 6.1 March Week 3 2025 dispatch plot (MANDATORY)  [HIGH]
- Run MPC with chosen H over March 2025 (we have data, even though optimization scope is Apr+Sep)
- Pick March 16-22 2025 (Week 3, Mon-Sun)
- 5-panel plot: load, PV, P_battery, P_grid, SoC vs time
- Save: `outputs/plots/march_week3_dispatch.png`

### 6.2 Results table  [HIGH]
| Metric                         | April 2025 | September 2025 | Total      |
|--------------------------------|-----------:|---------------:|-----------:|
| Baseline A (existing)          | -9.35      | +1.79          | **-7.57**  |
| Baseline B (no battery)        | +25.87     | +42.87         | +68.74     |
| Our controller (H=X)           |   ?        |   ?            |   ?        |
| Oracle (perfect foresight)     |   ?        |   ?            |   ?        |
| Savings vs A (€ / %)           |   ?        |   ?            |   ?        |
| Oracle gap (€ / %)             |   ?        |   ?            |   ?        |

### 6.3 NRMSE table  [HIGH]
| Set                     | RMSE (kW) | MAE (kW) | NRMSE (%) |
|-------------------------|----------:|---------:|----------:|
| Validation (Apr+Sep 24) |   ?       |   ?      |   ?       |
| Test (Apr+Sep 25)       |   ?       |   ?      |   ?       |
| Surprise (Day 2)        |   ?       |   ?      |   ?       |

### 6.4 Code packaging  [HIGH]
- All source under `src/`
- Reproducibility: `requirements.txt` complete, all seeds set
- Push final commit to GitHub
- Submit: GitHub URL or ZIP

---

## PHASE 7 — DAY 2 SURPRISE DATASET (~30 min, time-boxed)

### 7.1 At 13:00 Day 2: receive surprise dataset  [HIGH]
- Format: same as 2024/2025 but unseen residential site
- Action: run forecast model **WITHOUT retraining**
- Compute features using same pipeline (will work because no site-specific features)
- Compute RMSE / MAE / NRMSE on surprise dataset
- Save to `outputs/surprise_results.csv`

### 7.2 Update slide 5 with surprise NRMSE  [HIGH]

---

## PHASE 8 — PRESENTATION (Day 2, ~1h before submission)

### Slide 1 — Forecasting model
- Algorithm: ensemble of LightGBM + XGBoost + CatBoost
- Key features: lag_96, lag_672, Fourier(hour), tariff_band, weather (HDD/CDD)
- Validation: Apr+Sep 2024 holdout (mirrors test season)
- Numbers: RMSE, MAE, NRMSE on Apr+Sep 2025

### Slide 2 — Controller
- Method: rolling-horizon MPC, LP-relaxed (no binaries)
- Solver: HiGHS via CVXPY
- Horizon H = X (justified by Phase 4 knee analysis)
- One-sentence economic intuition: "discharge during F2 evenings, charge during F3 nights or PV surplus"

### Slide 3 — March Week 3 dispatch plot
- 5-panel: load / PV / P_battery / P_grid / SoC
- Annotate: F2 evening discharge events, midday PV charging

### Slide 4 — Results
- The table from 6.2
- Headline: "EUR X savings vs Baseline A (Y%) on Apr+Sep 2025"

### Slide 5 — Generalization
- Compare 2025 NRMSE vs surprise NRMSE
- One-sentence comment: "Models held within Z% — no site-specific overfit"

### Slide 6 — Hardest problem + improvements
- Real candidate problem: SoC reconstruction from corrupted 2025 battery data
- Or: choosing H — LP relaxation made the 10-horizon sweep tractable
- "With one more day": train PV forecast jointly, try Transformer

---

## TIME-BUDGET SUMMARY

| Phase                                         | Hours  | Day |
|-----------------------------------------------|--------|-----|
| 1 — Feature engineering                       | 1.5    | 1   |
| 2 — Forecasting (LGB+XGB+Cat ensemble)        | 2.5    | 1   |
| 3 — Optimizer LP + oracle + first MPC run     | 2.0    | 1/2 |
| 4 — Horizon sensitivity (10 horizons)         | 1.5    | 2   |
| 5 — Generalization checks                     | 0.5    | 1   |
| 6 — Deliverables (plots, tables, code)        | 1.5    | 2   |
| 7 — Surprise dataset                          | 0.5    | 2   |
| 8 — Presentation                              | 1.0    | 2   |
| Buffer                                        | 1.0    | 2   |
| **Total**                                     | **12** |     |

---

## RISKS & MITIGATIONS

| Risk                                              | Mitigation                                          |
|---------------------------------------------------|-----------------------------------------------------|
| LP infeasibility on edge cases                    | Add tiny slack variable on power balance, penalize  |
| Forecast NRMSE > 50% (poor)                       | Fall back to lag_96 persistence + scale by hour     |
| Surprise dataset has different schema             | Schema-detect, drop missing columns, use defaults   |
| MPC loop too slow                                 | Profile; switch CVXPY → direct scipy.optimize.linprog |
| Day 2 PV forecast required (not just actuals)     | Train tiny PV model on cloud_cover + hour + month   |
| GitHub push fails on submission day               | Have ZIP backup; pre-test push 30 min before deadline |

---

## FILE / MODULE LAYOUT

```
src/
├── data/
│   ├── loader.py            (DONE)
│   ├── build_processed.py   (DONE)
│   └── compute_baselines.py (DONE)
├── features/
│   ├── weather_fetch.py     (Phase 1.1)
│   └── build_features.py    (Phase 1.4)
├── models/
│   ├── lgbm_model.py        (Phase 2.2)
│   ├── xgb_model.py         (Phase 2.3)
│   ├── cat_model.py         (Phase 2.4)
│   ├── tune_optuna.py       (Phase 2.5)
│   └── ensemble.py          (Phase 2.6)
├── controller/
│   ├── lp_optimizer.py      (Phase 3.1)
│   ├── oracle.py            (Phase 3.2)
│   ├── mpc_loop.py          (Phase 3.3)
│   └── horizon_sweep.py     (Phase 4)
└── eval/
    ├── compute_bill.py      (use Baseline A formula)
    ├── metrics.py           (RMSE, MAE, NRMSE)
    └── plots.py             (March Week 3, horizon curve)

data/
├── raw/                  (DONE)
├── processed/            (DONE)
├── external/             (Phase 1.1 — Sondrio weather)
├── features/             (Phase 1.4)
└── forecasts/            (Phase 2.8)

outputs/
├── plots/                (existing 18 + new from Phase 4 & 6)
├── reports/              (DONE)
├── surprise_results.csv  (Phase 7)
└── horizon_sweep.csv     (Phase 4)
```

---

## SCORING TARGETS (100 base + 5 bonus)

| Component                          | Pts | Our target                         |
|------------------------------------|----:|------------------------------------|
| Controller savings vs Baseline A   |  35 | Beat -7.57 by EUR 5–15             |
| Forecasting NRMSE 2025             |  25 | NRMSE < 45%                        |
| Generalization NRMSE surprise      |  25 | NRMSE within 10% of 2025 NRMSE     |
| Reasoning & presentation           |  15 | Tight 6-slide deck, clear plots    |
| Extension 1 (horizon sensitivity)  |  +5 | 10 horizons + knee analysis        |
| **TOTAL**                          | 105 |                                    |

---

*Last updated: 2026-05-09. Edit freely as we progress.*
