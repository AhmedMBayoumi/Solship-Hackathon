"""
Modal: CatBoost with Optuna. ~20 min cap.
CatBoost handles categorical tariff_enc, dow, month natively.
"""
import modal

app = modal.App("solship-catboost")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4", "catboost==1.2.7",
        "optuna==3.6.1", "pyarrow==16.1.0",
    )
)


@app.function(image=image, timeout=1800)
def train(train_parquet: bytes, val_parquet: bytes, test_parquet: bytes, n_trials: int = 40) -> dict:
    import io, json, time
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor, Pool
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
    CAT_COLS = ["hour","dow","month","is_weekend","is_holiday","tariff_enc"]
    TARGET = "load_kw"

    train_df = pd.read_parquet(io.BytesIO(train_parquet))
    val_df   = pd.read_parquet(io.BytesIO(val_parquet))
    test_df  = pd.read_parquet(io.BytesIO(test_parquet))
    avail    = [c for c in FEATURE_COLS if c in train_df.columns]
    cat_idx  = [avail.index(c) for c in CAT_COLS if c in avail]

    def make_pool(df, target=True):
        X = df[avail].values
        y = df[TARGET].values if target else None
        return Pool(X, y, cat_features=cat_idx)

    tr_pool = make_pool(train_df)
    va_pool = make_pool(val_df)
    te_pool = make_pool(test_df, target=False)

    y_va = val_df[TARGET].values

    def nrmse(y_t, y_p):
        return float(np.sqrt(np.mean((y_t - y_p)**2)) / np.mean(y_t) * 100)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Features: {len(avail)} | Cat: {len(cat_idx)}")

    def objective(trial):
        params = {
            "iterations":        trial.suggest_int("iterations", 300, 2000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth":             trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1, 20),
            "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "random_seed": 42, "verbose": False, "allow_writing_files": False,
            "early_stopping_rounds": 50,
        }
        m = CatBoostRegressor(**params)
        m.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        return nrmse(y_va, m.predict(va_pool))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=1200)
    elapsed = time.time() - t0

    best = study.best_params
    print(f"\nOptuna: {len(study.trials)} trials in {elapsed:.0f}s | Best NRMSE: {study.best_value:.3f}%")

    # Retrain on train+val
    import pandas as _pd
    full_df = _pd.concat([train_df, val_df], ignore_index=True)
    full_pool = make_pool(full_df)
    final = CatBoostRegressor(**{**best, "verbose": False, "allow_writing_files": False, "random_seed": 42})
    final.fit(full_pool)

    val_preds  = final.predict(va_pool)
    test_preds = final.predict(te_pool) if len(test_df) else np.array([])
    val_nrmse  = nrmse(y_va, val_preds)

    print(f"Final CatBoost Val NRMSE: {val_nrmse:.3f}%")
    return {
        "best_params": best, "best_val_nrmse": study.best_value,
        "final_val_nrmse": val_nrmse,
        "final_val_rmse": float(np.sqrt(np.mean((y_va - val_preds)**2))),
        "final_val_mae": float(np.mean(np.abs(y_va - val_preds))),
        "val_preds": val_preds.tolist(),
        "test_preds": test_preds.tolist() if len(test_preds) else [],
        "val_timestamps": val_df["timestamp"].astype(str).tolist(),
        "test_timestamps": test_df["timestamp"].astype(str).tolist() if len(test_df) else [],
    }


@app.local_entrypoint()
def main():
    import json
    import pandas as pd
    from pathlib import Path

    ROOT = Path(__file__).parents[1]
    FEAT = ROOT / "data" / "features"

    result = train.remote(
        (FEAT / "features_train.parquet").read_bytes(),
        (FEAT / "features_val.parquet").read_bytes(),
        (FEAT / "features_test.parquet").read_bytes(),
        n_trials=40,
    )

    out_dir = ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "catboost_result.json", "w") as f:
        json.dump(result, f, indent=2)

    pred_dir = ROOT / "outputs" / "forecasts"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if result["test_preds"]:
        pd.DataFrame({"timestamp": result["test_timestamps"], "load_pred": result["test_preds"]}) \
          .to_csv(pred_dir / "catboost_test_preds.csv", index=False)

    print(f"Best: {result['best_val_nrmse']:.3f}%  Final: {result['final_val_nrmse']:.3f}%")
