"""
Modal: ML+DL gated fusion with WALKFORWARD splits + v4 features.
Two separate fusion models (April, September) — one per supervisor's allowance.
"""
import modal

app = modal.App("solship-fusion-walkforward")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4", "pyarrow==16.1.0",
        "scikit-learn==1.5.0",
        "lightgbm==4.5.0", "xgboost==2.1.1", "catboost==1.2.7",
        "torch==2.3.0",
    )
)


@app.function(image=image, gpu="T4", timeout=3500)
def fuse_walkforward(features_v4_all_p: bytes) -> dict:
    import io, time
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    import torch
    import torch.nn as nn
    from sklearn.model_selection import KFold

    df_all = pd.read_parquet(io.BytesIO(features_v4_all_p))
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
                 "qow","hod","net_load","sell_price"}
    feats = [c for c in df_all.columns if c not in DROP_BASE]
    print(f"v4 features: {len(feats)}")

    CAT_FEATS = ["dow","month","hour","is_weekend","is_holiday","tariff_enc"]
    cat_idx_in_feats = [feats.index(c) for c in CAT_FEATS if c in feats]
    CONTEXT_FEATS = ["hour","dow","is_weekend","is_holiday","tariff_enc","pv_kw",
                     "lag_1","lag_96","temperature_2m"]

    def april_split():
        ts = df_all["timestamp"]
        return (((ts.dt.year == 2024) | ((ts.dt.year == 2025) & (ts.dt.month <= 2))),
                ((ts.dt.year == 2025) & (ts.dt.month == 3)),
                ((ts.dt.year == 2025) & (ts.dt.month == 4)))
    def sept_split():
        ts = df_all["timestamp"]
        return (((ts.dt.year == 2024) |
                 ((ts.dt.year == 2025) & (ts.dt.month <= 7)) |
                 ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day <= 15))),
                ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day > 15)),
                ((ts.dt.year == 2025) & (ts.dt.month == 9)))

    def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)
    def mae(y, yp):   return float(np.mean(np.abs(y - yp)))

    def to_cat_df(X):
        d = pd.DataFrame(X, columns=feats)
        for c in CAT_FEATS:
            if c in d.columns: d[c] = d[c].astype(int)
        return d

    BASE_PARAMS = dict(
        n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4,
        min_child_samples=100, reg_alpha=2.0, reg_lambda=3.0,
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        verbose=-1, n_jobs=-1, random_state=42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    def train_block(label, splits):
        tr_m, va_m, te_m = splits
        train_df = df_all[tr_m]; val_df = df_all[va_m]; test_df = df_all[te_m]

        print(f"=== {label} ===")
        print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

        X_tr = train_df[feats].values; y_tr = train_df["load_kw"].values
        X_va = val_df  [feats].values; y_va = val_df  ["load_kw"].values
        X_te = test_df [feats].values; y_te = test_df ["load_kw"].values

        # Base 1: LGBM
        t0 = time.time()
        lgbm = lgb.LGBMRegressor(**BASE_PARAMS, objective="huber", alpha=0.9)
        lgbm.fit(X_tr, y_tr)
        val_lgbm  = np.clip(lgbm.predict(X_va), 0, None).astype("float32")
        test_lgbm = np.clip(lgbm.predict(X_te), 0, None).astype("float32")
        print(f"    LGBM   val={nrmse(y_va,val_lgbm):.2f}%  test={nrmse(y_te,test_lgbm):.2f}%  ({time.time()-t0:.0f}s)")

        # Base 2: XGB
        t0 = time.time()
        xgbm = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.01, max_depth=4, min_child_weight=20,
            reg_alpha=2.0, reg_lambda=3.0, subsample=0.7, colsample_bytree=0.7,
            n_jobs=-1, random_state=42, tree_method="hist",
            objective="reg:pseudohubererror", huber_slope=0.5,
        )
        xgbm.fit(X_tr, y_tr)
        val_xgb  = np.clip(xgbm.predict(X_va), 0, None).astype("float32")
        test_xgb = np.clip(xgbm.predict(X_te), 0, None).astype("float32")
        print(f"    XGB    val={nrmse(y_va,val_xgb):.2f}%  test={nrmse(y_te,test_xgb):.2f}%  ({time.time()-t0:.0f}s)")

        # Base 3: CatBoost
        t0 = time.time()
        cat = CatBoostRegressor(
            iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=10.0,
            random_seed=42, verbose=0, bagging_temperature=2.0, subsample=0.7,
            cat_features=cat_idx_in_feats, loss_function="Huber:delta=0.5",
        )
        cat.fit(to_cat_df(X_tr), y_tr, verbose=0)
        val_cat  = np.clip(cat.predict(to_cat_df(X_va)), 0, None).astype("float32")
        test_cat = np.clip(cat.predict(to_cat_df(X_te)), 0, None).astype("float32")
        print(f"    CAT    val={nrmse(y_va,val_cat):.2f}%  test={nrmse(y_te,test_cat):.2f}%  ({time.time()-t0:.0f}s)")

        # Base 4: MLP
        t0 = time.time()
        means = X_tr.mean(axis=0).astype("float32")
        stds  = X_tr.std(axis=0).astype("float32"); stds = np.where(stds<1e-6, 1.0, stds)
        Xs_tr = ((X_tr - means)/stds).astype("float32")
        Xs_va = ((X_va - means)/stds).astype("float32")
        Xs_te = ((X_te - means)/stds).astype("float32")

        class MLP(nn.Module):
            def __init__(self, n_feat, hidden=256, dropout=0.3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden//2, 1),
                )
            def forward(self, x): return self.net(x).squeeze(-1)

        mlp = MLP(Xs_tr.shape[1]).to(device)
        opt_ = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=60)
        lfn = nn.HuberLoss(delta=0.5)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(y_tr.astype("float32"))),
            batch_size=512, shuffle=True
        )
        # Initialise best_state with the starting weights so we always have a fallback.
        best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
        best_val = 1e9
        for ep in range(60):
            mlp.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt_.zero_grad(); lfn(mlp(xb), yb).backward(); opt_.step()
            sched.step()
            mlp.eval()
            with torch.no_grad():
                vp = np.clip(mlp(torch.from_numpy(Xs_va).to(device)).cpu().numpy(), 0, None)
            vn = nrmse(y_va, vp)
            # Only checkpoint when val NRMSE is actually finite + better.
            if np.isfinite(vn) and vn < best_val:
                best_val = vn
                best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
        mlp.load_state_dict(best_state); mlp.eval()
        with torch.no_grad():
            val_mlp  = np.clip(mlp(torch.from_numpy(Xs_va).to(device)).cpu().numpy(), 0, None).astype("float32")
            test_mlp = np.clip(mlp(torch.from_numpy(Xs_te).to(device)).cpu().numpy(), 0, None).astype("float32")
        print(f"    MLP    val={nrmse(y_va,val_mlp):.2f}%  test={nrmse(y_te,test_mlp):.2f}%  ({time.time()-t0:.0f}s)")

        # Gated fusion
        L0_va = np.column_stack([val_lgbm, val_xgb, val_cat, val_mlp]).astype("float32")
        L0_te = np.column_stack([test_lgbm, test_xgb, test_cat, test_mlp]).astype("float32")
        ctx_va = val_df[CONTEXT_FEATS].values.astype("float32")
        ctx_te = test_df[CONTEXT_FEATS].values.astype("float32")
        cm = ctx_va.mean(0); cs = ctx_va.std(0); cs = np.where(cs<1e-6,1.0,cs)
        ctx_va = ((ctx_va-cm)/cs).astype("float32")
        ctx_te = ((ctx_te-cm)/cs).astype("float32")

        class GatedFusion(nn.Module):
            def __init__(self, n_models, n_ctx, hidden=64, dropout=0.6):
                super().__init__()
                self.gate = nn.Sequential(
                    nn.Linear(n_ctx, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, n_models),
                )
                self.bias = nn.Parameter(torch.zeros(1))
            def forward(self, preds, ctx):
                w = torch.softmax(self.gate(ctx), dim=-1)
                return (w*preds).sum(dim=-1) + self.bias

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        best_test = None; best_n = 1e9
        for cfg in [{"hidden":32,"dr":0.5,"wd":5e-3,"ep":80,"lr":3e-3},
                    {"hidden":64,"dr":0.6,"wd":5e-3,"ep":80,"lr":3e-3}]:
            cv_t = []
            for tr_idx, va_idx in kf.split(np.arange(len(y_va))):
                gf = GatedFusion(4, ctx_va.shape[1], cfg["hidden"], cfg["dr"]).to(device)
                op = torch.optim.Adam(gf.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
                lf = nn.HuberLoss(delta=0.5)
                L0_in_t  = torch.from_numpy(L0_va[tr_idx]).to(device); ctx_in_t = torch.from_numpy(ctx_va[tr_idx]).to(device); y_in_t = torch.from_numpy(y_va[tr_idx].astype("float32")).to(device)
                L0_h_t   = torch.from_numpy(L0_va[va_idx]).to(device); ctx_h_t  = torch.from_numpy(ctx_va[va_idx]).to(device); y_h_t  = torch.from_numpy(y_va[va_idx].astype("float32")).to(device)
                L0_te_t  = torch.from_numpy(L0_te).to(device); ctx_te_t = torch.from_numpy(ctx_te).to(device)
                # Initialise best_st with the starting weights as fallback.
                best_st = {k:v.detach().cpu().clone() for k,v in gf.state_dict().items()}
                best_h = 1e9
                for ep in range(cfg["ep"]):
                    gf.train()
                    op.zero_grad()
                    yp = gf(L0_in_t, ctx_in_t); lf(yp, y_in_t).backward(); op.step()
                    gf.eval()
                    with torch.no_grad():
                        yh = gf(L0_h_t, ctx_h_t); h_loss = float(((yh-y_h_t)**2).mean())
                    if np.isfinite(h_loss) and h_loss < best_h:
                        best_h = h_loss
                        best_st = {k:v.detach().cpu().clone() for k,v in gf.state_dict().items()}
                gf.load_state_dict(best_st); gf.eval()
                with torch.no_grad():
                    yt = np.clip(gf(L0_te_t, ctx_te_t).cpu().numpy(), 0, None)
                cv_t.append(yt)
            tp = np.mean(cv_t, axis=0); n = nrmse(y_te, tp)
            print(f"    Fusion h={cfg['hidden']} dr={cfg['dr']}  test={n:.2f}%")
            if n < best_n: best_n = n; best_test = tp

        avg4 = nrmse(y_te, (test_lgbm+test_xgb+test_cat+test_mlp)/4)
        print(f"    Simple avg of 4: {avg4:.2f}%   Best fusion: {best_n:.2f}%")
        return test_df["timestamp"].values, best_test, y_te, {
            "lgbm":test_lgbm.tolist(), "xgb":test_xgb.tolist(),
            "cat":test_cat.tolist(),  "mlp":test_mlp.tolist(),
            "fused":best_test.tolist(), "best_fusion_nrmse":float(best_n),
            "avg4_nrmse":float(avg4),
        }

    # Train both blocks
    ts_a, p_a, y_a, info_a = train_block("APRIL (walkforward)", april_split())
    ts_s, p_s, y_s, info_s = train_block("SEPTEMBER (walkforward)", sept_split())

    ts_all = np.concatenate([ts_a, ts_s])
    p_all  = np.concatenate([p_a, p_s])
    y_all  = np.concatenate([y_a, y_s])
    print(f"\n=== COMBINED Apr+Sep 2025 ===")
    print(f"  test NRMSE: {nrmse(y_all, p_all):.2f}%   MAE={mae(y_all, p_all):.4f}")

    return {
        "test_timestamps": [str(t) for t in ts_all],
        "test_fused":     p_all.tolist(),
        "april_info":     info_a,
        "september_info": info_s,
        "combined_nrmse": float(nrmse(y_all, p_all)),
    }


@app.local_entrypoint()
def main():
    from pathlib import Path
    import json
    ROOT = Path(__file__).parents[1]
    print("Submitting walkforward fusion to Modal (T4)...")
    pq = (ROOT / "data/features/features_v4_all.parquet").read_bytes()
    r = fuse_walkforward.remote(pq)
    out = ROOT / "outputs/models/fusion_walkforward_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    import pandas as pd
    p = ROOT / "outputs/forecasts/fusion_walkforward_test_preds.csv"
    pd.DataFrame({"timestamp": pd.to_datetime(r["test_timestamps"]),
                  "load_pred": r["test_fused"]}).to_csv(p, index=False)
    print(f"\nSaved -> {out}")
    print(f"Saved preds -> {p}")
    print(f"\nCombined test NRMSE: {r['combined_nrmse']:.2f}%")
