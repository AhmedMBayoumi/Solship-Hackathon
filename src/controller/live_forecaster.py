"""
Live inference forecaster for rolling-horizon MPC.

Uses actual observed load history (including pre-month context) so that
lag_96, lag_672, and rolling features are always populated from real data.

For horizon steps beyond k=0, lags that fall inside the unobserved future
use recursively predicted values.
"""
import numpy as np
import pickle
from pathlib import Path

ROOT = Path(__file__).parents[2]

FIXED_HOLIDAYS = {(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)}
EASTER_MONDAYS = {2024:(4,1), 2025:(4,21)}
LOCAL_HOLIDAYS = {(6,19)}


def _is_holiday(ts):
    md = (ts.month, ts.day)
    em = EASTER_MONDAYS.get(ts.year)
    return int(md in FIXED_HOLIDAYS or md in LOCAL_HOLIDAYS or (em and md == em))


class LiveForecaster:
    """Load forecaster that maintains full causal history across the MPC loop."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = ROOT / "outputs" / "models" / "lgbm_model.pkl"
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        self.model  = obj["model"]
        self.fcols  = obj["feature_cols"]
        self._n_feat = len(self.fcols)
        self._col_idx = {c: i for i, c in enumerate(self.fcols)}

    def make_context(self, df_full, df_month):
        """
        Build the combined context arrays needed for live forecasting.

        Returns a dict with pre-computed arrays, plus the month offset.
        """
        start_ts   = df_month["timestamp"].iloc[0]
        full_idx   = df_full.index[df_full["timestamp"] == start_ts]
        if len(full_idx) == 0:
            # Fall back: assume month starts at position 0 in full
            month_offset = 0
        else:
            month_offset = int(full_idx[0])

        n_full = len(df_full)

        return {
            "load_full"  : df_full["load_kw"].values.astype(float),
            "pv_full"    : df_full["pv_kw"].values.astype(float),
            "net_full"   : (df_full["load_kw"] - df_full["pv_kw"]).values.astype(float),
            "temp_full"  : df_full["temperature_2m"].values.astype(float)  if "temperature_2m"  in df_full else np.full(n_full, 15.0),
            "rad_full"   : df_full["shortwave_radiation"].values.astype(float) if "shortwave_radiation" in df_full else np.zeros(n_full),
            "month_offset": month_offset,
            "df_month"   : df_month,
        }

    def forecast(self, ctx, t_local, H):
        """
        Forecast H steps starting at local index t_local in ctx["df_month"].

        ctx        : result of make_context()
        t_local    : current step index within df_month (0-based)
        H          : forecast horizon

        Returns np.ndarray shape (H_eff,).
        """
        df_month    = ctx["df_month"]
        offset      = ctx["month_offset"]      # where month starts in df_full
        load_full   = ctx["load_full"]
        pv_full     = ctx["pv_full"]
        net_full    = ctx["net_full"]
        temp_full   = ctx["temp_full"]
        rad_full    = ctx["rad_full"]
        n_full      = len(load_full)
        n_month     = len(df_month)

        H_eff = min(H, n_month - t_local)
        # Global index in df_full corresponding to t_local
        g0    = offset + t_local   # = global index of current step

        # We have actual load for global indices 0 .. g0-1
        # For indices >= g0, we'll accumulate predictions
        preds = []   # preds[k] = predicted load at global index g0+k

        def _actual_or_pred(g_idx):
            """Get load at global index g_idx (actual if < g0, else from preds)."""
            if g_idx < 0 or g_idx >= n_full + H_eff:
                return 0.0  # out of range — use 0
            if g_idx < g0:
                return float(load_full[g_idx])
            pred_k = g_idx - g0
            if pred_k < len(preds):
                return float(preds[pred_k])
            return 0.0

        def _roll_mean_std(center_g, window):
            """Mean and std of load over global indices [center_g-window, center_g-1]."""
            arr = np.array(
                [_actual_or_pred(center_g - window + i) for i in range(window)],
                dtype=float
            )
            return float(arr.mean()), float(arr.std() if window > 1 else 0.0)

        feat_buf = np.empty(self._n_feat, dtype=float)

        for k in range(H_eff):
            g = g0 + k                  # global index of step we're forecasting
            step_local = t_local + k
            row = df_month.iloc[step_local]

            # Lag values
            lag1   = _actual_or_pred(g - 1)
            lag4   = _actual_or_pred(g - 4)
            lag8   = _actual_or_pred(g - 8)
            lag96  = _actual_or_pred(g - 96)
            lag192 = _actual_or_pred(g - 192)
            lag672 = _actual_or_pred(g - 672)

            # Rolling stats (on load)
            r4m,  r4s  = _roll_mean_std(g, 4)
            r16m, _    = _roll_mean_std(g, 16)
            r96m, r96s = _roll_mean_std(g, 96)

            # PV lags — always actual
            pv_lag1  = float(pv_full[g - 1])  if g >= 1  else 0.0
            pv_lag96 = float(pv_full[g - 96]) if g >= 96 else 0.0

            # Net-load lags (use actual net_load series for indices in actual range)
            def _net_lag(lag):
                gi = g - lag
                if gi < 0:
                    return 0.0
                if gi < g0:
                    return float(net_full[gi])
                # In predicted range: use predicted load - actual pv
                pred_load = _actual_or_pred(gi)
                pv_val = float(pv_full[gi]) if gi < n_full else 0.0
                return pred_load - pv_val

            nl1  = _net_lag(1)
            nl96 = _net_lag(96)

            # Calendar
            ts     = row["timestamp"]
            hour   = int(ts.hour)
            minute = int(ts.minute)
            dow    = int(ts.dayofweek)
            month  = int(ts.month)
            doy    = int(ts.day_of_year)
            is_wknd = int(dow >= 5)
            is_hol  = _is_holiday(ts)
            tariff_enc = int(row.get("tariff_enc", 0))
            buy_price  = float(row.get("buy_price",  0.254))

            # Fourier (t_min = hour*60 + minute)
            t_min  = hour * 60 + minute
            sin_24h = np.sin(2 * np.pi * t_min / 1440)
            cos_24h = np.cos(2 * np.pi * t_min / 1440)
            sin_12h = np.sin(2 * np.pi * t_min / 720)
            cos_12h = np.cos(2 * np.pi * t_min / 720)
            sin_8h  = np.sin(2 * np.pi * t_min / 480)
            cos_8h  = np.cos(2 * np.pi * t_min / 480)
            sin_ann = np.sin(2 * np.pi * doy / 365.25)
            cos_ann = np.cos(2 * np.pi * doy / 365.25)

            # Weather (always actual for the evaluation set)
            temp  = float(row.get("temperature_2m",       15.0))
            rad   = float(row.get("shortwave_radiation",   0.0))
            cloud = float(row.get("cloud_cover",          50.0))
            rh    = float(row.get("relative_humidity_2m", 60.0))
            hdd   = float(row.get("hdd",  0.0))
            cdd   = float(row.get("cdd",  0.0))

            # Weather lags — use pre-computed full arrays
            temp_lag96 = float(temp_full[g - 96]) if g >= 96 else temp
            rad_lag96  = float(rad_full[g - 96])  if g >= 96 else rad

            feat_map = {
                "lag_1": lag1, "lag_4": lag4, "lag_8": lag8,
                "lag_96": lag96, "lag_192": lag192, "lag_672": lag672,
                "pv_lag1": pv_lag1, "pv_lag96": pv_lag96,
                "roll_4_mean": r4m, "roll_16_mean": r16m, "roll_96_mean": r96m,
                "roll_4_std": r4s, "roll_96_std": r96s,
                "net_load_lag1": nl1, "net_load_lag96": nl96,
                "hour": hour, "dow": dow, "month": month,
                "day_of_year": doy, "is_weekend": is_wknd,
                "is_holiday": is_hol, "tariff_enc": tariff_enc,
                "buy_price": buy_price,
                "sin_24h": sin_24h, "cos_24h": cos_24h,
                "sin_12h": sin_12h, "cos_12h": cos_12h,
                "sin_8h": sin_8h, "cos_8h": cos_8h,
                "sin_annual": sin_ann, "cos_annual": cos_ann,
                "temperature_2m": temp, "shortwave_radiation": rad,
                "cloud_cover": cloud, "relative_humidity_2m": rh,
                "hdd": hdd, "cdd": cdd,
                "temp_lag96": temp_lag96, "rad_lag96": rad_lag96,
            }

            for i, c in enumerate(self.fcols):
                v = feat_map.get(c, 0.0)
                feat_buf[i] = v if v == v else 0.0   # NaN check

            pred = float(self.model.predict(feat_buf.reshape(1, -1))[0])
            pred = max(0.0, pred)
            preds.append(pred)

        return np.array(preds, dtype=float)


def make_live_factory(df_full, model_path=None):
    """
    Returns a forecast_fn_factory compatible with run_both_months.

    df_full: full 2025 DataFrame with all feature columns (from features_test.parquet
             or the full features DataFrame).  Must include load_kw, pv_kw,
             temperature_2m, shortwave_radiation columns.
    """
    forecaster = LiveForecaster(model_path=model_path)

    def factory(df_month):
        ctx = forecaster.make_context(df_full, df_month)

        def forecast_fn(t, H):
            return forecaster.forecast(ctx, t, H)

        return forecast_fn

    return factory
