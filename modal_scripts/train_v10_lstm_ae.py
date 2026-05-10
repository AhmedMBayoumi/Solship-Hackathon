"""
MODAL v10 — User's DL feature-extractor idea (TRUE PyTorch LSTM autoencoder).

Pipeline (runs on Modal T4 GPU):
  1. Train an LSTM autoencoder on 2024 windows of [load,pv,temp,radiation].
     Encoder: LSTM 4 -> 32 -> bottleneck 16
     Decoder: 16 -> LSTM 32 -> 4   (reconstruct)
  2. Encode every row's past-96 window -> 16-dim bottleneck features.
  3. Concat with v7 tabular features -> v10 features (153 + 16 = 169).
  4. Train 8-bag LGBM walkforward (April: train through Mar 2025;
     September: train through Aug 2025).
  5. Return test predictions for both folds.

Local entrypoint reads features_v7_all.parquet and saves test preds to
outputs/forecasts/v10_lstm_ae_test_preds.csv.
"""
import modal

app   = modal.App("solship-v10-lstm-ae")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4", "pyarrow==16.1.0",
        "scikit-learn==1.5.0", "lightgbm==4.5.0",
        "torch==2.4.0",
    )
)
# Persistent volume so the result survives local disconnects.
vol = modal.Volume.from_name("solship-v10-results", create_if_missing=True)


@app.function(image=image, gpu="T4", cpu=4.0, memory=16384, timeout=3600,
              volumes={"/out": vol})
def train_v10(features_v7_all_p: bytes) -> dict:
    import io, time
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import torch
    import torch.nn as nn

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}", flush=True)

    df = pd.read_parquet(io.BytesIO(features_v7_all_p))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"v7 shape: {df.shape}", flush=True)

    WINDOW = 96
    N_BOTT = 16
    N_HID  = 32
    SIGNALS = ["load_kw", "pv_kw", "temperature_2m", "shortwave_radiation"]

    sig_arr = df[SIGNALS].ffill().bfill().fillna(0).values.astype("float32")
    mask_2024 = (df["timestamp"].dt.year == 2024).values
    chan_mean = sig_arr[mask_2024].mean(axis=0)
    chan_std  = sig_arr[mask_2024].std(axis=0) + 1e-6
    sig_z = (sig_arr - chan_mean) / chan_std

    # Build (n, window, n_chan) windows
    n = len(df)
    print(f"Building {n} past-{WINDOW} windows of {SIGNALS}...", flush=True)
    windows = np.zeros((n, WINDOW, len(SIGNALS)), dtype="float32")
    for t in range(WINDOW, n):
        windows[t] = sig_z[t-WINDOW:t]

    # ─── LSTM Autoencoder ────────────────────────────────────────────
    class LSTMAE(nn.Module):
        def __init__(self, n_chan=4, n_hid=N_HID, n_bott=N_BOTT, window=WINDOW):
            super().__init__()
            self.window = window; self.n_hid = n_hid
            self.enc_lstm = nn.LSTM(n_chan, n_hid, batch_first=True)
            self.enc_lin  = nn.Linear(n_hid, n_bott)
            self.dec_lin  = nn.Linear(n_bott, n_hid)
            self.dec_lstm = nn.LSTM(n_hid, n_hid, batch_first=True)
            self.dec_out  = nn.Linear(n_hid, n_chan)

        def encode(self, x):
            _, (h, _) = self.enc_lstm(x)
            return self.enc_lin(h.squeeze(0))

        def forward(self, x):
            z   = self.encode(x)
            rep = z.unsqueeze(1).repeat(1, self.window, 1)
            rep = self.dec_lin(rep)
            h0  = self.dec_lin(z).unsqueeze(0)
            c0  = torch.zeros_like(h0)
            out, _ = self.dec_lstm(rep, (h0, c0))
            return self.dec_out(out)

    # Train on 2024 only, subsampled hourly
    train_idx = np.where(mask_2024 & (np.arange(n) >= WINDOW))[0][::4]
    X_train = torch.from_numpy(windows[train_idx]).float()
    print(f"AE training windows: {len(X_train)}", flush=True)

    torch.manual_seed(42)
    model = LSTMAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    EPOCHS, BATCH = 12, 512
    print(f"Training LSTM-AE for {EPOCHS} epochs, batch={BATCH}...", flush=True)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X_train))
        losses = []
        for i in range(0, len(X_train), BATCH):
            xb = X_train[perm[i:i+BATCH]].to(DEVICE)
            yh = model(xb)
            loss = loss_fn(yh, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"  epoch {ep+1}/{EPOCHS}  recon_mse={np.mean(losses):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ─── Encode every row ───────────────────────────────────────────
    print("Encoding all rows...", flush=True)
    model.eval()
    bott = np.zeros((n, N_BOTT), dtype="float32")
    with torch.no_grad():
        for i in range(WINDOW, n, 1024):
            xb = torch.from_numpy(windows[i:i+1024]).float().to(DEVICE)
            bott[i:i+len(xb)] = model.encode(xb).cpu().numpy()
    ae_cols = [f"ae_{i:02d}" for i in range(N_BOTT)]
    for i, c in enumerate(ae_cols):
        df[c] = bott[:, i]
    df.loc[df.index < WINDOW, ae_cols] = np.nan
    print(f"  added {N_BOTT} AE features", flush=True)

    # ─── 8-bag LGBM walkforward ─────────────────────────────────────
    DROP_BASE = {"timestamp","load_kw","p_battery_kw","grid_kw","tariff_band","minute",
                 "qow","hod","net_load","sell_price"}
    feats = [c for c in df.columns if c not in DROP_BASE]
    print(f"v10 features: {len(feats)}", flush=True)

    ts = df["timestamp"]
    def april_split():
        return (((ts.dt.year == 2024) | ((ts.dt.year == 2025) & (ts.dt.month <= 2))),
                ((ts.dt.year == 2025) & (ts.dt.month == 3)),
                ((ts.dt.year == 2025) & (ts.dt.month == 4)))
    def sept_split():
        return (((ts.dt.year == 2024) |
                 ((ts.dt.year == 2025) & (ts.dt.month <= 7)) |
                 ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day <= 15))),
                ((ts.dt.year == 2025) & (ts.dt.month == 8) & (ts.dt.day > 15)),
                ((ts.dt.year == 2025) & (ts.dt.month == 9)))
    def nrmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)) / np.mean(y) * 100)

    LIGHT_CONFIGS = [
        {"num_leaves":63,"max_depth":8,"learning_rate":0.02,"min_child_samples":20,
         "reg_alpha":0.1,"reg_lambda":0.1,"subsample":0.9,"colsample_bytree":0.9},
        {"num_leaves":47,"max_depth":7,"learning_rate":0.015,"min_child_samples":30,
         "reg_alpha":0.3,"reg_lambda":0.5,"subsample":0.85,"colsample_bytree":0.85},
        {"num_leaves":95,"max_depth":10,"learning_rate":0.025,"min_child_samples":15,
         "reg_alpha":0.05,"reg_lambda":0.1,"subsample":0.85,"colsample_bytree":0.85},
        {"num_leaves":31,"max_depth":6,"learning_rate":0.025,"min_child_samples":40,
         "reg_alpha":0.5,"reg_lambda":0.5,"subsample":0.85,"colsample_bytree":0.85},
    ]

    def run(label, splits, n_bags=8, n_trees=2000):
        tr_m, va_m, te_m = splits
        tv_df = pd.concat([df[tr_m], df[va_m]], ignore_index=True).dropna(subset=feats).reset_index(drop=True)
        te_df = df[te_m].dropna(subset=feats).reset_index(drop=True)
        print(f"\n=== {label}  train+val={len(tv_df)}  test={len(te_df)} ===", flush=True)
        X_tv, y_tv = tv_df[feats].values, tv_df["load_kw"].values
        X_te, y_te = te_df[feats].values, te_df["load_kw"].values
        preds = np.zeros((n_bags, len(X_te)))
        importance = np.zeros(len(feats))
        t0 = time.time()
        for i in range(n_bags):
            cfg = dict(LIGHT_CONFIGS[i % len(LIGHT_CONFIGS)])
            seed = 42 + i
            cfg.update({"n_estimators":n_trees,"subsample_freq":1,"objective":"huber",
                        "alpha":0.9,"verbose":-1,"n_jobs":-1,"random_state":seed})
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(X_tv), size=int(len(X_tv)*0.9))
            m = lgb.LGBMRegressor(**cfg); m.fit(X_tv[idx], y_tv[idx])
            preds[i] = np.clip(m.predict(X_te), 0, None)
            importance += m.feature_importances_ / n_bags
            print(f"  bag {i+1}/{n_bags} ({time.time()-t0:.0f}s)", flush=True)
        avg = preds.mean(axis=0)
        n_te = nrmse(y_te, avg)
        print(f"  {label} NRMSE: {n_te:.2f}%", flush=True)
        ae_imp = sorted([(f, importance[feats.index(f)]) for f in ae_cols], key=lambda x: -x[1])
        print("  Top AE feature importances:")
        for f, im in ae_imp[:5]: print(f"    {f}: {im:.0f}")
        return te_df["timestamp"].values, avg, y_te, n_te

    ts_a, p_a, y_a, n_a = run("APRIL",     april_split())
    ts_s, p_s, y_s, n_s = run("SEPTEMBER", sept_split())

    ts_all = np.concatenate([ts_a, ts_s])
    p_all  = np.concatenate([p_a, p_s])
    y_all  = np.concatenate([y_a, y_s])
    n_combined = nrmse(y_all, p_all)

    print("\n" + "="*70, flush=True)
    print(f"  v10 (LSTM-AE features) combined NRMSE: {n_combined:.2f}%", flush=True)
    print(f"  vs prev best (online_retraining): 60.65%", flush=True)
    delta = n_combined - 60.65
    verdict = "WORSE" if delta > 0 else "IMPROVEMENT!" if delta < 0 else "tied"
    print(f"  delta vs baseline: {delta:+.2f}pp  ({verdict})", flush=True)

    result = {
        "test_timestamps":  [str(t) for t in ts_all],
        "test_predictions": p_all.tolist(),
        "april_nrmse":      float(n_a),
        "september_nrmse":  float(n_s),
        "combined_nrmse":   float(n_combined),
    }
    # Persist to volume so we can recover even if the local caller disconnects
    import json
    with open("/out/v10_result.json", "w") as f:
        json.dump(result, f)
    pd.DataFrame({"timestamp": ts_all, "load_pred": p_all}).to_csv("/out/v10_test_preds.csv", index=False)
    vol.commit()
    print("[volume] result saved to /out/v10_result.json + v10_test_preds.csv", flush=True)
    return result


@app.local_entrypoint()
def main():
    from pathlib import Path
    import json, pandas as pd
    ROOT = Path(__file__).parents[1]
    pq = (ROOT / "data/features/features_v7_all.parquet").read_bytes()
    print("Submitting v10 (LSTM-AE feature extractor) to Modal T4 GPU (spawn mode)...")
    # Use spawn so the function keeps running on Modal even if local disconnects.
    fc = train_v10.spawn(pq)
    print(f"Spawned function call: {fc.object_id}")
    print("If your local connection drops, recover the result with:")
    print("  modal volume get solship-v10-results /v10_result.json outputs/models/v10_lstm_ae_result.json")
    print("  modal volume get solship-v10-results /v10_test_preds.csv outputs/forecasts/v10_lstm_ae_test_preds.csv")
    r = fc.get()
    out_json = ROOT / "outputs/models/v10_lstm_ae_result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(r, indent=2))
    out_csv = ROOT / "outputs/forecasts/v10_lstm_ae_test_preds.csv"
    pd.DataFrame({"timestamp": pd.to_datetime(r["test_timestamps"]),
                  "load_pred": r["test_predictions"]}).to_csv(out_csv, index=False)
    print(f"\nv10 NRMSE: {r['combined_nrmse']:.2f}%")
    print(f"April: {r['april_nrmse']:.2f}%   September: {r['september_nrmse']:.2f}%")
    print(f"Saved -> {out_csv}")
