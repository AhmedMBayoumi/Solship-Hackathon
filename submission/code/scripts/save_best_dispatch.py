"""Save the H=672 LightGBM dispatch (our submission baseline) for plots."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from src.controller.mpc_loop import run_both_months
from src.eval.compute_bill import compute_bill

ROOT = Path(__file__).parents[1]
df_2025 = pd.read_csv(ROOT / "data/processed/dataset_processed.csv", parse_dates=["timestamp"])
df_2025 = df_2025[df_2025["timestamp"].dt.year == 2025].copy().reset_index(drop=True)

preds = pd.read_csv(ROOT / "outputs/forecasts/lgbm_test_preds.csv", parse_dates=["timestamp"])

def make_factory(preds_df):
    def factory(df_month):
        merged = df_month[["timestamp"]].merge(preds_df, on="timestamp", how="left")
        load_pred = merged["load_pred"].values.astype(float)
        for i in range(len(load_pred)):
            if np.isnan(load_pred[i]):
                load_pred[i] = load_pred[i-1] if i > 0 else 1.0
        def fn(t, H_in):
            end = min(t + H_in, len(load_pred))
            return load_pred[t:end]
        return fn
    return factory

H = int(sys.argv[1]) if len(sys.argv) > 1 else 96

t0 = time.time()
res, total = run_both_months(df_2025, make_factory(preds), H=H, verbose=False, label=f"H={H}")
elapsed = time.time() - t0

bills_per_month = {}
for m, name in [(4, "April"), (9, "September")]:
    sub = res[res["timestamp"].dt.month == m]
    b = compute_bill(sub, sub["p_grid_kw"])
    bills_per_month[name] = b

out = ROOT / "outputs" / f"mpc_dispatch_H{H}.parquet"
res.to_parquet(out, index=False)

print(f"H={H}  Total: EUR {total['net_bill']:+.2f}  vs A: EUR {total['net_bill']-(-7.57):+.2f}  ({elapsed:.0f}s)")
print(f"  April:     EUR {bills_per_month['April']['net_bill']:+.2f}")
print(f"  September: EUR {bills_per_month['September']['net_bill']:+.2f}")
print(f"  Saved -> {out}")
