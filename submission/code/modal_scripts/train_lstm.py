"""
Modal training: LSTM seq2seq for 96-step load forecast.
Tesla T4 GPU. Max 25 min training.

Input  : sequence of last 96 timesteps × 12 features
Output : next 96 timesteps of load_kw
"""
import modal

app = modal.App("solship-lstm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4",
        "torch==2.3.0", "scikit-learn==1.5.0", "pyarrow==16.1.0",
    )
)

GPU_TYPE = "T4"


@app.function(image=image, gpu=GPU_TYPE, timeout=1700)
def train_lstm(train_parquet: bytes, val_parquet: bytes, test_parquet: bytes,
               full_parquet: bytes, epochs: int = 30) -> dict:
    import io, json, time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Sequence features: keep things normalised + compact
    SEQ_FEATS = [
        "load_kw", "pv_kw",
        "sin_24h", "cos_24h", "sin_12h", "cos_12h",
        "sin_annual", "cos_annual",
        "is_weekend", "is_holiday", "tariff_enc",
        "buy_price",
    ]
    TARGET = "load_kw"
    L_PAST = 96     # 1-day history
    L_FUT  = 96     # 1-day forecast

    full_df  = pd.read_parquet(io.BytesIO(full_parquet)).sort_values("timestamp").reset_index(drop=True)
    train_df = pd.read_parquet(io.BytesIO(train_parquet))
    val_df   = pd.read_parquet(io.BytesIO(val_parquet))
    test_df  = pd.read_parquet(io.BytesIO(test_parquet))

    avail = [c for c in SEQ_FEATS if c in full_df.columns]
    print(f"Sequence features: {len(avail)} -> {avail}")

    # Standardisation (fit on train only)
    means = train_df[avail].mean().values.astype("float32")
    stds  = train_df[avail].std().replace(0, 1).values.astype("float32")

    full_arr = ((full_df[avail].values - means) / stds).astype("float32")
    targ_full = full_df[TARGET].values.astype("float32")
    full_ts  = full_df["timestamp"].values

    # Build sequences indexed by global position t (predict t .. t+L_FUT-1 from t-L_PAST..t-1)
    def build_seqs(target_df: pd.DataFrame):
        idx_in_full = full_df.index[full_df["timestamp"].isin(target_df["timestamp"])].to_numpy()
        valid = idx_in_full[(idx_in_full >= L_PAST) & (idx_in_full + L_FUT <= len(full_df))]
        n = len(valid)
        X = np.empty((n, L_PAST, len(avail)), dtype="float32")
        Y = np.empty((n, L_FUT), dtype="float32")
        for k, t in enumerate(valid):
            X[k] = full_arr[t - L_PAST : t]
            Y[k] = targ_full[t : t + L_FUT]
        return X, Y, valid

    print("Building sequences...")
    Xtr, Ytr, _ = build_seqs(train_df)
    Xva, Yva, _ = build_seqs(val_df)
    Xte, Yte, te_idx = build_seqs(test_df)
    te_targ_ts = full_ts[te_idx]   # timestamps where forecast STARTS
    print(f"  train seqs: {len(Xtr)}  val: {len(Xva)}  test: {len(Xte)}")

    class Seq2SeqLSTM(nn.Module):
        def __init__(self, n_feat, hidden=128, n_layers=2, dropout=0.2, out_steps=L_FUT):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, n_layers,
                                batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_steps),
            )
        def forward(self, x):
            out, _ = self.lstm(x)        # (B, L, H)
            last = out[:, -1, :]          # (B, H)
            return self.head(last)        # (B, L_FUT)

    model = Seq2SeqLSTM(n_feat=len(avail)).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    BATCH = 256
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=BATCH, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=BATCH, shuffle=False
    )

    best_val = 1e9
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb)
                va_loss += loss_fn(yp, yb).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)
        sched.step()
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 2 == 0 or ep == epochs - 1:
            print(f"  ep {ep:2d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  best={best_val:.4f}  elapsed={time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    model.eval()

    def predict_all(X):
        ds = TensorDataset(torch.from_numpy(X))
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
        out = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device)
                out.append(model(xb).cpu().numpy())
        return np.concatenate(out, axis=0)

    val_preds  = predict_all(Xva)   # (n_val, L_FUT)
    test_preds = predict_all(Xte)   # (n_test, L_FUT)

    def nrmse(y, yp):
        return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)

    val_nrmse_step1   = nrmse(Yva[:, 0], val_preds[:, 0])
    val_nrmse_horizon = nrmse(Yva, val_preds)
    print(f"Val NRMSE step-1 : {val_nrmse_step1:.2f}%")
    print(f"Val NRMSE 96-step: {val_nrmse_horizon:.2f}%")

    return {
        "val_preds":   val_preds.tolist(),
        "test_preds":  test_preds.tolist(),
        "val_nrmse_step1":   val_nrmse_step1,
        "val_nrmse_horizon": val_nrmse_horizon,
        "test_start_timestamps": [str(t) for t in te_targ_ts],
        "feat_means": means.tolist(),
        "feat_stds":  stds.tolist(),
        "feat_cols":  avail,
        "L_PAST": L_PAST,
        "L_FUT":  L_FUT,
    }


@app.local_entrypoint()
def main():
    from pathlib import Path
    import json
    ROOT = Path(__file__).parents[1]
    tr_b = (ROOT / "data/features/features_train.parquet").read_bytes()
    va_b = (ROOT / "data/features/features_val.parquet").read_bytes()
    te_b = (ROOT / "data/features/features_test.parquet").read_bytes()
    fl_b = (ROOT / "data/features/features_all.parquet").read_bytes()

    print("Submitting LSTM training to Modal (T4 GPU)...")
    r = train_lstm.remote(tr_b, va_b, te_b, fl_b, epochs=30)

    out = ROOT / "outputs/models/lstm_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"Saved -> {out}")
    print(f"Val NRMSE step-1: {r['val_nrmse_step1']:.2f}%")
    print(f"Val NRMSE 96-step: {r['val_nrmse_horizon']:.2f}%")
