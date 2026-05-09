import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / float(np.mean(np.array(y_true))) * 100


def report(y_true, y_pred, label=""):
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    n = nrmse(y_true, y_pred)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RMSE={r:.4f} kW  MAE={m:.4f} kW  NRMSE={n:.2f}%")
    return {"rmse": r, "mae": m, "nrmse": n}
