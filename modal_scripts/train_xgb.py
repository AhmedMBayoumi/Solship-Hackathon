"""
Modal training script: XGBoost with Optuna.
Max 25 min on Modal (well under 2-hour rule).
"""
import modal

app = modal.App("solship-xgb")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4", "xgboost==2.1.1",
        "scikit-learn==1.5.0", "optuna==3.6.1", "pyarrow==16.1.0",
    )
)


@app.function(image=image, timeout=1800)
def train(train_parquet: bytes, val_parquet: bytes, test_parquet: bytes, n_trials: int = 60) -> dict:
    import io, json, time
    import numpy as np
    import pandas as pd
    import xgboost as xgb
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
    avail    = [c for c in FEATURE_COLS if c in train_df.columns]

    X_tr, y_tr = train_df[avail].values, train_df[TARGET].values
    X_va, y_va = val_df[avail].values,   val_df[TARGET].values
    X_te       = test_df[avail].values

    def nrmse(y_t, y_p):
        return float(np.sqrt(np.mean((y_t - y_p)**2)) / np.mean(y_t) * 100)

    print(f"Train: {len(X_tr)} | Val: {len(X_va)} | Features: {len(avail)}")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
            "eval_metric": "rmse", "early_stopping_rounds": 50,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return nrmse(y_va, model.predict(X_va))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    # Hard time cap 25 min
    study.optimize(objective, n_trials=n_trials, timeout=1500)
    elapsed = time.time() - t0

    best = study.best_params
    print(f"\nOptuna: {len(study.trials)} trials in {elapsed:.0f}s | Best NRMSE: {study.best_value:.3f}%")
    print(f"Best params: {json.dumps(best, indent=2)}")

    X_full = np.vstack([X_tr, X_va])
    y_full = np.hstack([y_tr, y_va])
    best_no_es = {k: v for k, v in best.items() if k not in ("early_stopping_rounds", "eval_metric")}
    final = xgb.XGBRegressor(**{**best_no_es, "verbosity": 0, "n_jobs": -1, "random_state": 42})
    final.fit(X_full, y_full)

    val_preds  = final.predict(X_va)
    test_preds = final.predict(X_te) if len(X_te) else np.array([])
    val_nrmse  = nrmse(y_va, val_preds)
    val_rmse   = float(np.sqrt(np.mean((y_va - val_preds)**2)))
    val_mae    = float(np.mean(np.abs(y_va - val_preds)))

    print(f"\n--- Final XGBoost ---")
    print(f"Val NRMSE: {val_nrmse:.3f}% | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")

    return {
        "best_params": best, "best_val_nrmse": study.best_value,
        "final_val_nrmse": val_nrmse, "final_val_rmse": val_rmse, "final_val_mae": val_mae,
        "val_preds": val_preds.tolist(),
        "test_preds": test_preds.tolist() if len(test_preds) else [],
        "val_timestamps": val_df["timestamp"].astype(str).tolist(),
        "test_timestamps": test_df["timestamp"].astype(str).tolist() if len(test_df) else [],
        "feature_cols": avail,
    }


@app.local_entrypoint()
def main():
    import json
    import pandas as pd
    from pathlib import Path

    ROOT = Path(__file__).parents[1]
    FEAT = ROOT / "data" / "features"

    print("Running XGBoost Optuna on Modal...")
    result = train.remote(
        (FEAT / "features_train.parquet").read_bytes(),
        (FEAT / "features_val.parquet").read_bytes(),
        (FEAT / "features_test.parquet").read_bytes(),
        n_trials=60,
    )

    out_dir = ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "xgb_result.json", "w") as f:
        json.dump(result, f, indent=2)

    pred_dir = ROOT / "outputs" / "forecasts"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if result["test_preds"]:
        pd.DataFrame({"timestamp": result["test_timestamps"], "load_pred": result["test_preds"]}) \
          .to_csv(pred_dir / "xgb_test_preds.csv", index=False)

    print(f"Best val NRMSE: {result['best_val_nrmse']:.3f}%  |  Final: {result['final_val_nrmse']:.3f}%")
