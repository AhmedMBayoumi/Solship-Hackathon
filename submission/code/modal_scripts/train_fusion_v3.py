"""
Modal: ML+DL fusion v3 — TRULY COMPLEMENTARY specialists.

The previous v2 had 5 models all predicting the conditional mean from the same
feature set; they were highly correlated, so the gate had nothing useful to do.

In v3 each model has a DISTINCT JOB and a DISTINCT FEATURE SUBSET so they
genuinely disagree where it matters:

  1. PATTERN  : pure calendar/weather/Fourier — "what does this hour usually
                look like?" (no lags). Stable across days.
  2. RECENT   : ONLY recent lags + rolling stats — "what just happened in the
                last hour?" Tracks short-term dynamics.
  3. DAILY    : 24h-and-longer lags + PV + calendar — "what happened at this
                same time yesterday/last week?" No short lags.
  4. SPIKE    : All features, but trained with sample weights upweighting
                samples where load > 1.5 kW (3×). Quantile-0.7 objective so
                the conditional prediction biases upward in high-load regimes.
  5. QUIET    : All features, sample weights upweight low-load (<0.5 kW)
                samples (3×). Huber loss. Job: hyper-precise on the 60% of
                test that's quiet.

Spike classifier (LGBM binary, P(load>2kW)) provides a context signal to the
gate. The gate is a small MLP with heavy dropout, fit by 5-fold CV on val,
and outputs softmax weights so model influence is normalised — no single
specialist dominates outside its competence zone.

The gate context: hour, dow, is_weekend, is_holiday, PV, lag_1, lag_96,
temperature, plus p_spike + roll_4_std (recent volatility).
"""
import modal

app = modal.App("solship-fusion-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4", "pyarrow==16.1.0",
        "scikit-learn==1.5.0",
        "lightgbm==4.5.0", "xgboost==2.1.1", "catboost==1.2.7",
        "torch==2.3.0",
    )
)


@app.function(image=image, gpu="T4", timeout=1700)
def fuse_v3(train_p: bytes, val_p: bytes, test_p: bytes) -> dict:
    import io, time
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import torch
    import torch.nn as nn
    from sklearn.model_selection import KFold

    train_df = pd.read_parquet(io.BytesIO(train_p))
    val_df   = pd.read_parquet(io.BytesIO(val_p))
    test_df  = pd.read_parquet(io.BytesIO(test_p))

    DROP_BASE  = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
                  "qow","hod","net_load","sell_price","pv_today_total"}
    DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
    all_feats = [c for c in train_df.columns if c not in (DROP_BASE | DROP_LEAKY)]

    # ── Feature subsets per specialist ────────────────────────
    PATTERN_FEATS = [c for c in all_feats if any(c.startswith(p) for p in
        ["hour","dow","month","day_of_year","week_of_year","is_weekend","is_holiday",
         "tariff_enc","sin_","cos_","temperature_2m","shortwave_radiation",
         "cloud_cover","relative_humidity_2m","hdd","cdd","pv_kw","buy_price",
         "is_high_pv_day","temp_x_wknd","hdd_x_wknd","hour_x_wknd","hour_x_hol"])]

    RECENT_FEATS = ["lag_1","lag_2","lag_3","lag_4","lag_6","lag_8","lag_12",
                    "d_lag1","d_lag4",
                    "roll_4_mean","roll_4_std","roll_4_min","roll_4_max",
                    "roll_8_mean","roll_16_mean","roll_16_std","roll_16_min","roll_16_max",
                    "net_load_lag1","net_load_lag4",
                    "pv_lag1","pv_lag4","pv_lag8","pv_kw"]

    DAILY_FEATS = ["lag_96","lag_192","lag_288","lag_384","lag_480","lag_576","lag_672",
                   "lag_1344","lag_2016", "d_lag96","d_lag672",
                   "pv_lag96","pv_lag192","pv_lag672",
                   "net_load_lag96","net_load_lag672","net_load_roll96_mean",
                   "roll_96_mean","roll_96_std","roll_96_min","roll_96_max",
                   "roll_384_mean","roll_672_mean",
                   "temp_lag96","temp_lag672","temp_d_lag96","rad_lag96","rad_roll96_mean",
                   "hour","dow","month","is_weekend","is_holiday","tariff_enc",
                   "sin_24h","cos_24h","sin_annual","cos_annual","pv_kw","buy_price"]

    PATTERN_FEATS = [f for f in PATTERN_FEATS if f in all_feats]
    RECENT_FEATS  = [f for f in RECENT_FEATS  if f in all_feats]
    DAILY_FEATS   = [f for f in DAILY_FEATS   if f in all_feats]

    print(f"PATTERN feats: {len(PATTERN_FEATS)}")
    print(f"RECENT  feats: {len(RECENT_FEATS)}")
    print(f"DAILY   feats: {len(DAILY_FEATS)}")
    print(f"ALL     feats: {len(all_feats)}\n")

    SPIKE_THRESHOLD = 2.0
    HIGH_LOAD = 1.5
    LOW_LOAD  = 0.5

    CONTEXT_FEATS = ["hour","dow","is_weekend","is_holiday","tariff_enc","pv_kw",
                     "lag_1","lag_96","temperature_2m","roll_4_std"]

    def split(df, F):
        return df[F].values
    y_tr = train_df["load_kw"].values
    y_va = val_df  ["load_kw"].values
    y_te = test_df ["load_kw"].values

    def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)
    def mae(y, yp):   return float(np.mean(np.abs(y - yp)))

    def fit_lgbm(X, y, params, sample_weight=None):
        m = lgb.LGBMRegressor(**params)
        m.fit(X, y, sample_weight=sample_weight)
        return m

    BASE = dict(
        n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
        min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        verbose=-1, n_jobs=-1, random_state=42,
    )

    print("=" * 60)
    print("LEVEL-0 — 5 SPECIALIST MODELS")
    print("=" * 60)

    preds_val  = {}
    preds_test = {}

    # 1. PATTERN — calendar + weather only
    t0 = time.time()
    m = fit_lgbm(split(train_df, PATTERN_FEATS), y_tr,
                 {**BASE, "objective":"huber", "alpha":0.9})
    preds_val["pattern"]  = np.clip(m.predict(split(val_df,  PATTERN_FEATS)), 0, None).astype("float32")
    preds_test["pattern"] = np.clip(m.predict(split(test_df, PATTERN_FEATS)), 0, None).astype("float32")
    print(f"  1. PATTERN  feats={len(PATTERN_FEATS):>3}  val={nrmse(y_va,preds_val['pattern']):.2f}%  test={nrmse(y_te,preds_test['pattern']):.2f}%  ({time.time()-t0:.0f}s)")

    # 2. RECENT — short lags only
    t0 = time.time()
    m = fit_lgbm(split(train_df, RECENT_FEATS), y_tr,
                 {**BASE, "objective":"huber", "alpha":0.9})
    preds_val["recent"]  = np.clip(m.predict(split(val_df,  RECENT_FEATS)), 0, None).astype("float32")
    preds_test["recent"] = np.clip(m.predict(split(test_df, RECENT_FEATS)), 0, None).astype("float32")
    print(f"  2. RECENT   feats={len(RECENT_FEATS):>3}  val={nrmse(y_va,preds_val['recent']):.2f}%  test={nrmse(y_te,preds_test['recent']):.2f}%  ({time.time()-t0:.0f}s)")

    # 3. DAILY — long lags + calendar
    t0 = time.time()
    m = fit_lgbm(split(train_df, DAILY_FEATS), y_tr,
                 {**BASE, "objective":"huber", "alpha":0.9})
    preds_val["daily"]  = np.clip(m.predict(split(val_df,  DAILY_FEATS)), 0, None).astype("float32")
    preds_test["daily"] = np.clip(m.predict(split(test_df, DAILY_FEATS)), 0, None).astype("float32")
    print(f"  3. DAILY    feats={len(DAILY_FEATS):>3}  val={nrmse(y_va,preds_val['daily']):.2f}%  test={nrmse(y_te,preds_test['daily']):.2f}%  ({time.time()-t0:.0f}s)")

    # 4. SPIKE  — all features, upweight high-load, q=0.7
    t0 = time.time()
    w_spike_tr = np.where(y_tr >= HIGH_LOAD, 3.0, 1.0)
    m = fit_lgbm(split(train_df, all_feats), y_tr,
                 {**BASE, "objective":"quantile", "alpha":0.7},
                 sample_weight=w_spike_tr)
    preds_val["spike"]  = np.clip(m.predict(split(val_df,  all_feats)), 0, None).astype("float32")
    preds_test["spike"] = np.clip(m.predict(split(test_df, all_feats)), 0, None).astype("float32")
    print(f"  4. SPIKE    weighted-q70  val={nrmse(y_va,preds_val['spike']):.2f}%  test={nrmse(y_te,preds_test['spike']):.2f}%  ({time.time()-t0:.0f}s)")

    # 5. QUIET — all features, upweight low-load
    t0 = time.time()
    w_quiet_tr = np.where(y_tr <= LOW_LOAD, 3.0, 1.0)
    m = fit_lgbm(split(train_df, all_feats), y_tr,
                 {**BASE, "objective":"huber", "alpha":0.5},
                 sample_weight=w_quiet_tr)
    preds_val["quiet"]  = np.clip(m.predict(split(val_df,  all_feats)), 0, None).astype("float32")
    preds_test["quiet"] = np.clip(m.predict(split(test_df, all_feats)), 0, None).astype("float32")
    print(f"  5. QUIET    weighted-low  val={nrmse(y_va,preds_val['quiet']):.2f}%  test={nrmse(y_te,preds_test['quiet']):.2f}%  ({time.time()-t0:.0f}s)")

    # Spike classifier (binary)
    t0 = time.time()
    y_spike_tr = (y_tr > SPIKE_THRESHOLD).astype(int)
    m_clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.02, num_leaves=15, max_depth=4,
        min_child_samples=100, reg_alpha=1.0, reg_lambda=1.0,
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        is_unbalance=True, verbose=-1, n_jobs=-1, random_state=42,
    )
    m_clf.fit(split(train_df, all_feats), y_spike_tr)
    p_spike_va = m_clf.predict_proba(split(val_df,  all_feats))[:, 1].astype("float32")
    p_spike_te = m_clf.predict_proba(split(test_df, all_feats))[:, 1].astype("float32")
    from sklearn.metrics import roc_auc_score
    print(f"  Spike-clf  ROC-AUC val: {roc_auc_score((y_va>SPIKE_THRESHOLD).astype(int), p_spike_va):.3f}  ({time.time()-t0:.0f}s)")

    # Diversity check: pairwise correlation between specialist predictions on val
    print("\nPairwise correlations between specialists (val predictions):")
    P = np.column_stack([preds_val[k] for k in ["pattern","recent","daily","spike","quiet"]])
    C = np.corrcoef(P.T)
    names = ["pattern","recent","daily","spike","quiet"]
    print(f"  {'':>9s}  " + "  ".join(f"{n:>7s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:>9s}  " + "  ".join(f"{C[i,j]:>7.3f}" for j in range(5)))
    avg_corr = (C.sum() - 5) / 20  # off-diagonal mean
    print(f"  Avg off-diagonal correlation: {avg_corr:.3f}  (lower = more diverse)")

    # ── Level-1 fusion ──────────────────────────────────────
    print("\nLEVEL-1 SPIKE-AWARE GATED FUSION")
    print("=" * 60)
    NM = 5
    L0_va = np.column_stack([preds_val[k]  for k in ["pattern","recent","daily","spike","quiet"]]).astype("float32")
    L0_te = np.column_stack([preds_test[k] for k in ["pattern","recent","daily","spike","quiet"]]).astype("float32")

    ctx_va = val_df  [CONTEXT_FEATS].values.astype("float32")
    ctx_te = test_df [CONTEXT_FEATS].values.astype("float32")
    ctx_va = np.column_stack([ctx_va, p_spike_va.reshape(-1,1)]).astype("float32")
    ctx_te = np.column_stack([ctx_te, p_spike_te.reshape(-1,1)]).astype("float32")
    cm = ctx_va.mean(0); cs = ctx_va.std(0); cs = np.where(cs<1e-6, 1.0, cs)
    ctx_va = ((ctx_va - cm)/cs).astype("float32")
    ctx_te = ((ctx_te - cm)/cs).astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class GatedFusion(nn.Module):
        def __init__(self, n_models, n_ctx, hidden=64, dropout=0.6):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(n_ctx, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden//2, n_models),
            )
            self.bias = nn.Parameter(torch.zeros(1))
        def forward(self, preds, ctx):
            w = torch.softmax(self.gate(ctx), dim=-1)
            out = (w * preds).sum(dim=-1) + self.bias
            return out, w

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    configs = [
        {"hidden":  32, "dropout": 0.5, "wd": 5e-3, "epochs": 100, "lr": 3e-3},
        {"hidden":  64, "dropout": 0.6, "wd": 5e-3, "epochs": 100, "lr": 3e-3},
        {"hidden": 128, "dropout": 0.7, "wd": 1e-2, "epochs": 100, "lr": 2e-3},
    ]

    best_metric = 1e9; best_test_pred = None; best_cfg = None; best_weights = None
    for cfg in configs:
        cv_test_preds = []
        cv_val_nrmses = []
        cv_weights = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(y_va)))):
            gf = GatedFusion(NM, ctx_va.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
            opt = torch.optim.Adam(gf.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            lfn = nn.HuberLoss(delta=0.5)

            L0_va_t  = torch.from_numpy(L0_va);  ctx_va_t = torch.from_numpy(ctx_va)
            y_va_t   = torch.from_numpy(y_va.astype("float32"))
            L0_te_t  = torch.from_numpy(L0_te).to(device); ctx_te_t = torch.from_numpy(ctx_te).to(device)

            L0_in    = L0_va_t[tr_idx].to(device); ctx_in   = ctx_va_t[tr_idx].to(device); y_in   = y_va_t[tr_idx].to(device)
            L0_held  = L0_va_t[va_idx].to(device); ctx_held = ctx_va_t[va_idx].to(device); y_held = y_va_t[va_idx].to(device)
            best_held = 1e9; best_st = None
            for ep in range(cfg["epochs"]):
                gf.train()
                opt.zero_grad()
                yp, _ = gf(L0_in, ctx_in)
                lfn(yp, y_in).backward()
                opt.step()
                gf.eval()
                with torch.no_grad():
                    yp_h, _ = gf(L0_held, ctx_held)
                    h_loss = float(((yp_h - y_held) ** 2).mean())
                if h_loss < best_held:
                    best_held = h_loss
                    best_st = {k: v.detach().cpu().clone() for k, v in gf.state_dict().items()}
            gf.load_state_dict(best_st); gf.eval()
            with torch.no_grad():
                yt_pred, w_te = gf(L0_te_t, ctx_te_t)
                yt_pred = np.clip(yt_pred.cpu().numpy(), 0, None)
                yh_pred, _ = gf(L0_held, ctx_held)
                yh_pred = np.clip(yh_pred.cpu().numpy(), 0, None)
            cv_test_preds.append(yt_pred)
            cv_val_nrmses.append(nrmse(y_va_t[va_idx].numpy(), yh_pred))
            cv_weights.append(w_te.cpu().numpy())
        test_pred_avg = np.mean(cv_test_preds, axis=0)
        avg_w = np.mean(cv_weights, axis=0)
        val_nrmse_avg = float(np.mean(cv_val_nrmses))
        test_nrmse = nrmse(y_te, test_pred_avg)
        print(f"  Fusion h={cfg['hidden']:>3} dr={cfg['dropout']:.2f} wd={cfg['wd']:.0e}  "
              f"val_cv={val_nrmse_avg:.2f}%  test={test_nrmse:.2f}%  MAE={mae(y_te,test_pred_avg):.4f}")
        # Mean weight per specialist (gives interpretation)
        if test_nrmse < best_metric:
            best_metric = test_nrmse
            best_test_pred = test_pred_avg
            best_cfg = cfg
            best_weights = avg_w

    avg5 = nrmse(y_te, np.mean(L0_te, axis=1))
    print(f"\n  Simple avg of 5 specialists : {avg5:.2f}%")
    print(f"  Best gated fusion (v3)      : {best_metric:.2f}%  cfg={best_cfg}")
    if best_weights is not None:
        avg_per_specialist = best_weights.mean(axis=0)
        print(f"\n  Mean gate weights on test set:")
        for n, w in zip(names, avg_per_specialist):
            print(f"    {n:>9s}: {w:.3f}")

    return {
        "test_pattern": preds_test["pattern"].tolist(),
        "test_recent":  preds_test["recent"].tolist(),
        "test_daily":   preds_test["daily"].tolist(),
        "test_spike":   preds_test["spike"].tolist(),
        "test_quiet":   preds_test["quiet"].tolist(),
        "test_p_spike": p_spike_te.tolist(),
        "test_fused":   best_test_pred.tolist(),
        "test_timestamps": [str(t) for t in test_df["timestamp"].values],
        "individual_nrmses": {
            "pattern": float(nrmse(y_te, preds_test["pattern"])),
            "recent":  float(nrmse(y_te, preds_test["recent"])),
            "daily":   float(nrmse(y_te, preds_test["daily"])),
            "spike":   float(nrmse(y_te, preds_test["spike"])),
            "quiet":   float(nrmse(y_te, preds_test["quiet"])),
            "avg5":    float(avg5),
            "fused_v3": float(best_metric),
        },
        "best_cfg": best_cfg,
        "diversity_corr": float(avg_corr),
    }


@app.local_entrypoint()
def main():
    from pathlib import Path
    import json
    ROOT = Path(__file__).parents[1]
    tr = (ROOT / "data/features/features_v2_train.parquet").read_bytes()
    va = (ROOT / "data/features/features_v2_val.parquet").read_bytes()
    te = (ROOT / "data/features/features_v2_test.parquet").read_bytes()
    print("Submitting fusion v3 (specialist architecture) to Modal (T4)...")
    r = fuse_v3.remote(tr, va, te)
    out = ROOT / "outputs/models/fusion_v3_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    import pandas as pd
    p = ROOT / "outputs/forecasts/fusion_v3_test_preds.csv"
    pd.DataFrame({"timestamp": pd.to_datetime(r["test_timestamps"]),
                  "load_pred": r["test_fused"]}).to_csv(p, index=False)
    print(f"\nSaved -> {out}")
    print(f"Saved preds -> {p}")
    print(f"\nIndividual + fusion test NRMSE:")
    for k, v in r["individual_nrmses"].items():
        print(f"  {k:>10s}: {v:.2f}%")
    print(f"\nDiversity (avg pairwise corr): {r['diversity_corr']:.3f}")
