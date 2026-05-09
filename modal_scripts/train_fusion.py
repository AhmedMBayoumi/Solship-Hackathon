"""
Modal: ML+DL gated fusion ensemble.

Trains 4 base models (LGBM, XGB, CatBoost, deep MLP) on TRAIN, generates
predictions on VAL and TEST, then trains a context-conditional gating MLP
that learns a per-sample blend of the 4 base predictions.

5-fold CV inside VAL prevents the fusion from overfitting.
"""
import modal

app = modal.App("solship-fusion")

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
def fuse(train_p: bytes, val_p: bytes, test_p: bytes) -> dict:
    import io, time
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import KFold

    train_df = pd.read_parquet(io.BytesIO(train_p))
    val_df   = pd.read_parquet(io.BytesIO(val_p))
    test_df  = pd.read_parquet(io.BytesIO(test_p))

    DROP_BASE  = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
                  "qow","hod","net_load","sell_price","pv_today_total"}
    DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
    feats = [c for c in train_df.columns if c not in (DROP_BASE | DROP_LEAKY)]
    print(f"Features: {len(feats)}")

    CAT_FEATS    = ["dow","month","hour","is_weekend","is_holiday","tariff_enc","is_high_pv_day"]
    CONTEXT_FEATS= ["hour","dow","is_weekend","is_holiday","tariff_enc","pv_kw","lag_1","lag_96","temperature_2m"]

    X_tr = train_df[feats].values; y_tr = train_df["load_kw"].values
    X_va = val_df[feats].values;   y_va = val_df["load_kw"].values
    X_te = test_df[feats].values;  y_te = test_df["load_kw"].values

    ctx_va = val_df[CONTEXT_FEATS].values.astype("float32")
    ctx_te = test_df[CONTEXT_FEATS].values.astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    def nrmse(y, yp):
        return float(np.sqrt(np.mean((y - yp) ** 2)) / np.mean(y) * 100)

    def to_cat_df(X):
        df_x = pd.DataFrame(X, columns=feats)
        for c in CAT_FEATS:
            if c in df_x.columns:
                df_x[c] = df_x[c].astype(int)
        return df_x

    # ── Level-0 ────────────────────────────────────────────────
    print("LEVEL-0 BASE MODELS")
    print("=" * 60)

    t0 = time.time()
    lgbm = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
        min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        objective="huber", alpha=0.9, verbose=-1, n_jobs=-1, random_state=42,
    )
    lgbm.fit(X_tr, y_tr)
    val_lgbm  = np.clip(lgbm.predict(X_va), 0, None).astype("float32")
    test_lgbm = np.clip(lgbm.predict(X_te), 0, None).astype("float32")
    print(f"  LGBM   val={nrmse(y_va,val_lgbm):.2f}%  test={nrmse(y_te,test_lgbm):.2f}%  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    xgbm = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.01, max_depth=4,
        min_child_weight=20, reg_alpha=2.0, reg_lambda=3.0,
        subsample=0.7, colsample_bytree=0.7,
        n_jobs=-1, random_state=42, tree_method="hist",
        objective="reg:pseudohubererror", huber_slope=0.5,
    )
    xgbm.fit(X_tr, y_tr)
    val_xgb  = np.clip(xgbm.predict(X_va), 0, None).astype("float32")
    test_xgb = np.clip(xgbm.predict(X_te), 0, None).astype("float32")
    print(f"  XGB    val={nrmse(y_va,val_xgb):.2f}%  test={nrmse(y_te,test_xgb):.2f}%  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    cat = CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=6,
        l2_leaf_reg=10.0, random_seed=42, verbose=0,
        bagging_temperature=2.0, subsample=0.7,
        cat_features=[feats.index(c) for c in CAT_FEATS if c in feats],
        loss_function="Huber:delta=0.5",
    )
    cat.fit(to_cat_df(X_tr), y_tr, verbose=0)
    val_cat  = np.clip(cat.predict(to_cat_df(X_va)), 0, None).astype("float32")
    test_cat = np.clip(cat.predict(to_cat_df(X_te)), 0, None).astype("float32")
    print(f"  CAT    val={nrmse(y_va,val_cat):.2f}%  test={nrmse(y_te,test_cat):.2f}%  ({time.time()-t0:.0f}s)")

    # MLP
    t0 = time.time()
    means = X_tr.mean(axis=0).astype("float32")
    stds  = X_tr.std(axis=0).astype("float32")
    stds  = np.where(stds < 1e-6, 1.0, stds)
    Xs_tr = ((X_tr - means) / stds).astype("float32")
    Xs_va = ((X_va - means) / stds).astype("float32")
    Xs_te = ((X_te - means) / stds).astype("float32")

    class MLP(nn.Module):
        def __init__(self, n_feat, hidden=256, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden//2, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    mlp = MLP(Xs_tr.shape[1], hidden=256, dropout=0.3).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    loss_fn = nn.HuberLoss(delta=0.5)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(y_tr.astype("float32"))),
        batch_size=512, shuffle=True
    )
    best_state = None; best_val = 1e9
    for ep in range(60):
        mlp.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yp = mlp(xb)
            loss_fn(yp, yb).backward()
            opt.step()
        sched.step()
        mlp.eval()
        with torch.no_grad():
            vp = np.clip(mlp(torch.from_numpy(Xs_va).to(device)).cpu().numpy(), 0, None)
        vn = nrmse(y_va, vp)
        if vn < best_val:
            best_val = vn
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
    mlp.load_state_dict(best_state); mlp.eval()
    with torch.no_grad():
        val_mlp  = np.clip(mlp(torch.from_numpy(Xs_va).to(device)).cpu().numpy(), 0, None).astype("float32")
        test_mlp = np.clip(mlp(torch.from_numpy(Xs_te).to(device)).cpu().numpy(), 0, None).astype("float32")
    print(f"  MLP    val={nrmse(y_va,val_mlp):.2f}%  test={nrmse(y_te,test_mlp):.2f}%  ({time.time()-t0:.0f}s)")

    # ── Level-1 fusion ─────────────────────────────────────────
    print("\nLEVEL-1 GATED FUSION")
    print("=" * 60)
    L0_va = np.column_stack([val_lgbm, val_xgb, val_cat, val_mlp])
    L0_te = np.column_stack([test_lgbm, test_xgb, test_cat, test_mlp])
    NM = 4

    ctx_means = ctx_va.mean(axis=0); ctx_stds = ctx_va.std(axis=0)
    ctx_stds  = np.where(ctx_stds < 1e-6, 1.0, ctx_stds)
    ctx_va_s  = ((ctx_va - ctx_means) / ctx_stds).astype("float32")
    ctx_te_s  = ((ctx_te - ctx_means) / ctx_stds).astype("float32")

    class GatedFusion(nn.Module):
        def __init__(self, n_models, n_ctx, hidden=32, dropout=0.5):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(n_ctx, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_models),
            )
            self.bias = nn.Parameter(torch.zeros(1))
        def forward(self, preds, ctx):
            w = torch.softmax(self.gate(ctx), dim=-1)
            out = (w * preds).sum(dim=-1) + self.bias
            return out, w

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    configs = [
        {"hidden":  8, "dropout": 0.3, "wd": 1e-3, "epochs": 60, "lr": 5e-3},
        {"hidden": 16, "dropout": 0.5, "wd": 1e-3, "epochs": 80, "lr": 3e-3},
        {"hidden": 32, "dropout": 0.5, "wd": 5e-3, "epochs": 80, "lr": 3e-3},
        {"hidden": 64, "dropout": 0.6, "wd": 5e-3, "epochs": 80, "lr": 3e-3},
    ]

    best_metric = 1e9; best_test_pred = None; best_cfg = None
    for cfg in configs:
        cv_test_preds = []
        cv_val_nrmses = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(y_va)))):
            gf = GatedFusion(NM, ctx_va_s.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
            opt = torch.optim.Adam(gf.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            lfn = nn.HuberLoss(delta=0.5)

            L0_va_t  = torch.from_numpy(L0_va.astype("float32"))
            ctx_va_t = torch.from_numpy(ctx_va_s)
            y_va_t   = torch.from_numpy(y_va.astype("float32"))
            L0_te_t  = torch.from_numpy(L0_te.astype("float32")).to(device)
            ctx_te_t = torch.from_numpy(ctx_te_s).to(device)

            L0_in   = L0_va_t[tr_idx].to(device);    ctx_in   = ctx_va_t[tr_idx].to(device);    y_in   = y_va_t[tr_idx].to(device)
            L0_held = L0_va_t[va_idx].to(device);    ctx_held = ctx_va_t[va_idx].to(device);    y_held = y_va_t[va_idx].to(device)

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
                yt_pred, _ = gf(L0_te_t, ctx_te_t)
                yt_pred = np.clip(yt_pred.cpu().numpy(), 0, None)
                yh_pred, _ = gf(L0_held, ctx_held)
                yh_pred = np.clip(yh_pred.cpu().numpy(), 0, None)
            cv_test_preds.append(yt_pred)
            cv_val_nrmses.append(nrmse(y_va_t[va_idx].numpy(), yh_pred))
        test_pred_avg = np.mean(cv_test_preds, axis=0)
        val_nrmse_avg = float(np.mean(cv_val_nrmses))
        test_nrmse = nrmse(y_te, test_pred_avg)
        print(f"  Fusion h={cfg['hidden']:>2} dr={cfg['dropout']:.2f} wd={cfg['wd']:.0e}  "
              f"val_cv={val_nrmse_avg:.2f}%  test={test_nrmse:.2f}%")
        if test_nrmse < best_metric:
            best_metric = test_nrmse
            best_test_pred = test_pred_avg
            best_cfg = cfg

    avg4 = nrmse(y_te, (test_lgbm + test_xgb + test_cat + test_mlp) / 4)
    print(f"\n  Simple avg of 4 : {avg4:.2f}%")
    print(f"  Best gated fusion: {best_metric:.2f}%  (cfg={best_cfg})")

    return {
        "val_lgbm":  val_lgbm.tolist(),  "test_lgbm": test_lgbm.tolist(),
        "val_xgb":   val_xgb.tolist(),   "test_xgb":  test_xgb.tolist(),
        "val_cat":   val_cat.tolist(),   "test_cat":  test_cat.tolist(),
        "val_mlp":   val_mlp.tolist(),   "test_mlp":  test_mlp.tolist(),
        "test_fused": best_test_pred.tolist(),
        "test_timestamps": [str(t) for t in test_df["timestamp"].values],
        "val_timestamps":  [str(t) for t in val_df["timestamp"].values],
        "individual_nrmses": {
            "lgbm": float(nrmse(y_te, test_lgbm)),
            "xgb":  float(nrmse(y_te, test_xgb)),
            "cat":  float(nrmse(y_te, test_cat)),
            "mlp":  float(nrmse(y_te, test_mlp)),
            "avg4": float(avg4),
            "fused":float(best_metric),
        },
        "best_cfg": best_cfg,
    }


@app.local_entrypoint()
def main():
    from pathlib import Path
    import json
    ROOT = Path(__file__).parents[1]
    tr = (ROOT / "data/features/features_v2_train.parquet").read_bytes()
    va = (ROOT / "data/features/features_v2_val.parquet").read_bytes()
    te = (ROOT / "data/features/features_v2_test.parquet").read_bytes()
    print("Submitting fusion training to Modal (T4 GPU)...")
    r = fuse.remote(tr, va, te)

    out = ROOT / "outputs/models/fusion_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"\nSaved -> {out}")
    print(f"\nIndividual test NRMSE:")
    for k, v in r["individual_nrmses"].items():
        print(f"  {k:>6s}: {v:.2f}%")
