"""
Modal training script: LightGBM with Optuna hyperparameter tuning.
Run: $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"; python -m modal run modal_scripts/train_lgbm.py

Streams results back to terminal. Saves best model params + predictions.
"""
import modal

app = modal.App("solship-lgbm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4", "lightgbm==4.6.0",
        "scikit-learn==1.5.0", "optuna==3.6.1", "pyarrow==16.1.0",
    )
)

# ── Mount project data and features ──────────────────────────────────────────
import os
from pathlib import Path

LOCAL_ROOT = Path(__file__).parents[1]

# We'll pass data as bytes to avoid large mounts
@app.function(image=image, timeout=1800)
def train(
    train_parquet: bytes,
    val_parquet:   bytes,
    test_parquet:  bytes,
    n_trials:      int = 80,
) -> dict:
    import io, json, time
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    FEATURE_COLS = [
        "lag_1","lag_4","lag_8","lag_96","lag_192","lag_672",
        "pv_lag1","pv_lag96",
        "roll_4_mean","roll_16_mean","roll_96_mean","roll_4_std","roll_96_std",
        "net_load_lag1","net_load_lag96",
        "hour","dow","month","day_of_year","is_weekend","is_holiday","tariff_enc",
        "buy_price",
        "sin_24h","cos_24h","sin_12h","cos_12h","sin_8h","cos_8h",
        "sin_annual","cos_annual",
        "temperature_2m","shortwave_radiation","cloud_cover","relative_humidity_2m",
        "hdd","cdd","temp_lag96","rad_lag96",
    ]
    TARGET = "load_kw"

    train_df = pd.read_parquet(io.BytesIO(train_parquet))
    val_df   = pd.read_parquet(io.BytesIO(val_parquet))
    test_df  = pd.read_parquet(io.BytesIO(test_parquet))

    # Filter to available cols (in case weather is missing)
    avail = [c for c in FEATURE_COLS if c in train_df.columns]

    X_tr = train_df[avail].values
    y_tr = train_df[TARGET].values
    X_va = val_df[avail].values
    y_va = val_df[TARGET].values
    X_te = test_df[avail].values

    def nrmse(y_t, y_p):
        return float(np.sqrt(np.mean((y_t - y_p)**2)) / np.mean(y_t) * 100)

    print(f"Train: {len(X_tr)} rows | Val: {len(X_va)} rows | Test: {len(X_te)} rows")
    print(f"Features: {len(avail)}")

    # ── Optuna objective ──────────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_va)
        return nrmse(y_va, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_params
    print(f"\nOptuna done: {n_trials} trials in {elapsed:.0f}s")
    print(f"Best val NRMSE: {study.best_value:.3f}%")
    print(f"Best params: {json.dumps(best, indent=2)}")

    # ── Retrain on full train+val with best params ────────────────────────
    X_full = np.vstack([X_tr, X_va])
    y_full = np.hstack([y_tr, y_va])
    final_model = lgb.LGBMRegressor(**{**best, "verbose": -1, "n_jobs": -1, "random_state": 42})
    final_model.fit(X_full, y_full)

    val_preds  = final_model.predict(X_va)
    test_preds = final_model.predict(X_te) if len(X_te) > 0 else np.array([])

    val_nrmse  = nrmse(y_va, val_preds)
    val_rmse   = float(np.sqrt(np.mean((y_va - val_preds)**2)))
    val_mae    = float(np.mean(np.abs(y_va - val_preds)))

    print(f"\n--- Final Model (retrained on train+val) ---")
    print(f"Val NRMSE : {val_nrmse:.3f}%")
    print(f"Val RMSE  : {val_rmse:.4f} kW")
    print(f"Val MAE   : {val_mae:.4f} kW")

    # Feature importance (top 20)
    fi = sorted(zip(avail, final_model.feature_importances_),
                key=lambda x: -x[1])[:20]
    print("\nTop 20 feature importances:")
    for fname, fimp in fi:
        bar = "█" * int(fimp / max(v for _,v in fi) * 40)
        print(f"  {fname:<25} {fimp:7.0f}  {bar}")

    return {
        "best_params": best,
        "best_val_nrmse": study.best_value,
        "final_val_nrmse": val_nrmse,
        "final_val_rmse": val_rmse,
        "final_val_mae": val_mae,
        "val_preds": val_preds.tolist(),
        "test_preds": test_preds.tolist() if len(test_preds) > 0 else [],
        "val_timestamps": val_df["timestamp"].astype(str).tolist(),
        "test_timestamps": test_df["timestamp"].astype(str).tolist() if len(test_df) > 0 else [],
        "feature_cols": avail,
        "feature_importance": {k: int(v) for k, v in fi},
    }


@app.local_entrypoint()
def main():
    import json, pickle
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import io

    ROOT = Path(__file__).parents[1]
    FEAT = ROOT / "data" / "features"

    print("Loading parquet files...")
    train_b = (FEAT / "features_train.parquet").read_bytes()
    val_b   = (FEAT / "features_val.parquet").read_bytes()
    test_b  = (FEAT / "features_test.parquet").read_bytes()

    print("Running LightGBM Optuna on Modal...")
    result = train.remote(train_b, val_b, test_b, n_trials=80)

    out_dir = ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "lgbm_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save predictions as CSV for the MPC loop
    if result["val_preds"]:
        pd.DataFrame({
            "timestamp": result["val_timestamps"],
            "load_pred": result["val_preds"],
        }).to_csv(ROOT / "outputs" / "forecasts" / "lgbm_val_preds.csv", index=False)

    if result["test_preds"]:
        pred_dir = ROOT / "outputs" / "forecasts"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "timestamp": result["test_timestamps"],
            "load_pred": result["test_preds"],
        }).to_csv(pred_dir / "lgbm_test_preds.csv", index=False)

    print(f"\nSaved results to {out_dir}")
    print(f"Best val NRMSE: {result['best_val_nrmse']:.3f}%")
    print(f"Final val NRMSE (retrained): {result['final_val_nrmse']:.3f}%")
    print(f"Best params: {json.dumps(result['best_params'], indent=2)}")
