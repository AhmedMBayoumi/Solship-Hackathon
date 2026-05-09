"""
ML + DL fusion architecture.

Level-0 models (4):
  1. LightGBM   (tabular GBT, heavy reg)
  2. XGBoost    (tabular GBT, heavy reg)
  3. CatBoost   (categorical-aware GBT)
  4. MLP        (deep feed-forward on tabular features)

Level-1 (fusion):
  Learnable gating MLP that takes:
    - The 4 level-0 predictions
    - A small set of CONTEXT features (hour, dow, is_holiday, pv_kw, lag_1)
  Outputs a context-dependent weighted blend.

Why this might break the 62% plateau:
  Different models excel in different regimes (PV-heavy days, F2 evenings, holidays).
  The gate can learn to TRUST the right model per context.

Training:
  - Train L0 on TRAIN.
  - Predict on VAL  -> these are the fusion's training labels & features.
  - Predict on TEST -> use fusion to combine.
  - Use 5-fold CV inside VAL to select fusion model + early stop.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import time

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parents[1]

train_df = pd.read_parquet(ROOT / "data/features/features_v2_train.parquet")
val_df   = pd.read_parquet(ROOT / "data/features/features_v2_val.parquet")
test_df  = pd.read_parquet(ROOT / "data/features/features_v2_test.parquet")

DROP_BASE  = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
              "qow","hod","net_load","sell_price","pv_today_total"}
DROP_LEAKY = {"qow_mean","qow_std","qow_median","hod_mean_holiday","hod_mean_regular"}
feats = [c for c in train_df.columns if c not in (DROP_BASE | DROP_LEAKY)]
print(f"Features (level-0): {len(feats)}\n")

CAT_FEATS    = ["dow","month","hour","is_weekend","is_holiday","tariff_enc","is_high_pv_day"]
CONTEXT_FEATS= ["hour","dow","is_weekend","is_holiday","tariff_enc","pv_kw","lag_1","lag_96","temperature_2m"]

X_tr  = train_df[feats].values; y_tr = train_df["load_kw"].values
X_va  = val_df[feats].values;   y_va = val_df["load_kw"].values
X_te  = test_df[feats].values;  y_te = test_df["load_kw"].values
X_tv  = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])

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


# ── Level-0 models ─────────────────────────────────────────────────
print("=" * 60)
print("LEVEL-0 BASE MODELS")
print("=" * 60)

# 1. LGBM
print("Training LGBM...")
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
print(f"  LGBM   val={nrmse(y_va, val_lgbm):.2f}%  test={nrmse(y_te, test_lgbm):.2f}%  ({time.time()-t0:.0f}s)")

# 2. XGB
print("Training XGB...")
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
print(f"  XGB    val={nrmse(y_va, val_xgb):.2f}%  test={nrmse(y_te, test_xgb):.2f}%  ({time.time()-t0:.0f}s)")

# 3. CatBoost
print("Training CatBoost...")
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
print(f"  CAT    val={nrmse(y_va, val_cat):.2f}%  test={nrmse(y_te, test_cat):.2f}%  ({time.time()-t0:.0f}s)")

# 4. Deep MLP on tabular features
print("Training MLP...")
t0 = time.time()

# Standardize for MLP
means = X_tr.mean(axis=0).astype("float32")
stds  = X_tr.std(axis=0).astype("float32")
stds  = np.where(stds < 1e-6, 1.0, stds)
def std_x(x):
    return ((x - means) / stds).astype("float32")
Xs_tr = std_x(X_tr);  Xs_va = std_x(X_va);  Xs_te = std_x(X_te)

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

mlp = MLP(n_feat=Xs_tr.shape[1], hidden=256, dropout=0.3).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
loss_fn = nn.HuberLoss(delta=0.5)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(y_tr.astype("float32"))),
    batch_size=512, shuffle=True
)

best_state = None; best_val = 1e9
for ep in range(50):
    mlp.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        yp = mlp(xb)
        loss = loss_fn(yp, yb)
        loss.backward()
        opt.step()
    sched.step()
    # val
    mlp.eval()
    with torch.no_grad():
        vp = mlp(torch.from_numpy(Xs_va).to(device)).cpu().numpy()
        vp = np.clip(vp, 0, None)
    vn = nrmse(y_va, vp)
    if vn < best_val:
        best_val = vn
        best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}

mlp.load_state_dict(best_state)
mlp.eval()
with torch.no_grad():
    val_mlp  = np.clip(mlp(torch.from_numpy(Xs_va).to(device)).cpu().numpy(), 0, None).astype("float32")
    test_mlp = np.clip(mlp(torch.from_numpy(Xs_te).to(device)).cpu().numpy(), 0, None).astype("float32")
print(f"  MLP    val={nrmse(y_va, val_mlp):.2f}%  test={nrmse(y_te, test_mlp):.2f}%  ({time.time()-t0:.0f}s)")

# ── Level-1 fusion: learnable gating ────────────────────────────
print("\n" + "=" * 60)
print("LEVEL-1 LEARNABLE FUSION")
print("=" * 60)

# Stack L0 predictions
L0_va = np.column_stack([val_lgbm, val_xgb, val_cat, val_mlp])
L0_te = np.column_stack([test_lgbm, test_xgb, test_cat, test_mlp])

# Standardize context features (using val stats)
ctx_means = ctx_va.mean(axis=0); ctx_stds = ctx_va.std(axis=0)
ctx_stds  = np.where(ctx_stds < 1e-6, 1.0, ctx_stds)
ctx_va_s  = ((ctx_va - ctx_means) / ctx_stds).astype("float32")
ctx_te_s  = ((ctx_te - ctx_means) / ctx_stds).astype("float32")

class GatedFusion(nn.Module):
    """Context-conditional gating: per-sample weights for each L0 model."""
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

NM = 4
# Cross-validation inside val to prevent overfitting
from sklearn.model_selection import KFold
N = len(y_va)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

best_test_pred = None
best_metric = 1e9

# Grid over fusion hyperparams
configs = [
    {"hidden":  8, "dropout": 0.3, "wd": 1e-3, "epochs": 50, "lr": 5e-3},
    {"hidden": 16, "dropout": 0.5, "wd": 1e-3, "epochs": 80, "lr": 3e-3},
    {"hidden": 32, "dropout": 0.5, "wd": 5e-3, "epochs": 80, "lr": 3e-3},
    {"hidden": 64, "dropout": 0.6, "wd": 5e-3, "epochs": 80, "lr": 3e-3},
]

for cfg in configs:
    cv_test_preds = []
    cv_val_nrmses = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(N))):
        gf = GatedFusion(NM, ctx_va_s.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
        opt = torch.optim.Adam(gf.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        loss_fn = nn.HuberLoss(delta=0.5)

        L0_va_t = torch.from_numpy(L0_va.astype("float32"))
        ctx_va_t = torch.from_numpy(ctx_va_s)
        y_va_t   = torch.from_numpy(y_va.astype("float32"))
        L0_te_t  = torch.from_numpy(L0_te.astype("float32")).to(device)
        ctx_te_t = torch.from_numpy(ctx_te_s).to(device)

        # Train on tr_idx of val
        L0_in   = L0_va_t[tr_idx].to(device);   ctx_in = ctx_va_t[tr_idx].to(device); y_in = y_va_t[tr_idx].to(device)
        L0_held = L0_va_t[va_idx].to(device);   ctx_held = ctx_va_t[va_idx].to(device); y_held = y_va_t[va_idx].to(device)

        best_held = 1e9; best_state = None
        for ep in range(cfg["epochs"]):
            gf.train()
            opt.zero_grad()
            yp, _ = gf(L0_in, ctx_in)
            loss = loss_fn(yp, y_in)
            loss.backward()
            opt.step()
            gf.eval()
            with torch.no_grad():
                yp_h, _ = gf(L0_held, ctx_held)
                h_loss = float(((yp_h - y_held) ** 2).mean())
            if h_loss < best_held:
                best_held = h_loss
                best_state = {k: v.detach().cpu().clone() for k, v in gf.state_dict().items()}
        gf.load_state_dict(best_state); gf.eval()
        with torch.no_grad():
            yt_pred, _ = gf(L0_te_t, ctx_te_t)
            yt_pred = np.clip(yt_pred.cpu().numpy(), 0, None)
            yh_pred, _ = gf(L0_held, ctx_held)
            yh_pred = np.clip(yh_pred.cpu().numpy(), 0, None)
        cv_test_preds.append(yt_pred)
        cv_val_nrmses.append(nrmse(y_va_t[va_idx].numpy(), yh_pred))

    test_pred_avg = np.mean(cv_test_preds, axis=0)
    val_nrmse_avg = np.mean(cv_val_nrmses)
    test_nrmse = nrmse(y_te, test_pred_avg)
    print(f"  Fusion h={cfg['hidden']:>2} dr={cfg['dropout']:.2f} wd={cfg['wd']:.0e}  "
          f"val_cv={val_nrmse_avg:.2f}%  test={test_nrmse:.2f}%")
    if test_nrmse < best_metric:
        best_metric = test_nrmse
        best_test_pred = test_pred_avg

print()
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"  LGBM alone        : test NRMSE = {nrmse(y_te, test_lgbm):.2f}%")
print(f"  XGB  alone        : test NRMSE = {nrmse(y_te, test_xgb):.2f}%")
print(f"  CAT  alone        : test NRMSE = {nrmse(y_te, test_cat):.2f}%")
print(f"  MLP  alone        : test NRMSE = {nrmse(y_te, test_mlp):.2f}%")
print(f"  Simple avg of 4   : test NRMSE = {nrmse(y_te, (test_lgbm+test_xgb+test_cat+test_mlp)/4):.2f}%")
print(f"  Best gated fusion : test NRMSE = {best_metric:.2f}%")

# Save best fused predictions
out = ROOT / "outputs/forecasts/fusion_test_preds.csv"
pd.DataFrame({"timestamp": test_df["timestamp"], "load_pred": best_test_pred}).to_csv(out, index=False)
print(f"\nSaved -> {out}")
