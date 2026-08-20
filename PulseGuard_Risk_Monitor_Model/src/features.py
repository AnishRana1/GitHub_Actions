import numpy as np
import pandas as pd
from src.determinism import configure_determinism
configure_determinism()

FEATURE_SCHEMA_VERSION = "pulseguard-14-feature-schema-v1"
CANONICAL_FEATURE_COLS = [
    "return_1d", "log_return", "hl_range_pct", "oc_return", "zscore",
    "rolling_volatility", "volatility_ratio", "volume_zscore",
    "momentum_5d", "momentum_20d", "trend_gap",
    "drawdown", "consec_down", "atr_pct",
]


def make_features(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 100,
    forward_horizon: int = 5,
    risk_threshold: float = 0.04,
    min_positive_events: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Build features and supervised labels.

    Labels are based on the worst DOWNSIDE move within forward_horizon days.
    The default risk_threshold is 0.04 (4% drop) — small enough that enough
    positive training examples exist across all timeframes and tickers.

    If the requested threshold produces fewer than min_positive_events positives,
    it is automatically relaxed to produce at least min_positive_events events
    (capped at 20% event rate so the model is not predicting a constant 1).
    """
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Feature input is missing required columns: {missing}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).sort_values("timestamp", kind="mergesort").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # --- Price-based features ---
    out["return_1d"] = out["close"].pct_change()
    out["log_return"] = np.log(out["close"]).diff()
    out["hl_range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_return"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    # --- Rolling statistics ---
    out["rolling_mean"] = out["close"].rolling(short_window, min_periods=short_window).mean()
    out["rolling_std"] = out["close"].rolling(short_window, min_periods=short_window).std()
    out["zscore"] = (out["close"] - out["rolling_mean"]) / out["rolling_std"].replace(0, np.nan)

    out["rolling_volatility"] = out["log_return"].rolling(short_window, min_periods=short_window).std()
    out["long_volatility"] = out["log_return"].rolling(long_window, min_periods=short_window).std()
    out["volatility_ratio"] = out["rolling_volatility"] / out["long_volatility"].replace(0, np.nan)

    volume_mean = out["volume"].rolling(short_window, min_periods=short_window).mean()
    volume_std = out["volume"].rolling(short_window, min_periods=short_window).std()
    out["volume_zscore"] = (out["volume"] - volume_mean) / volume_std.replace(0, np.nan)

    # --- Momentum & trend ---
    out["momentum_5d"] = out["close"] / out["close"].shift(5) - 1
    out["momentum_20d"] = out["close"] / out["close"].shift(20) - 1
    out["trend_mean"] = out["close"].rolling(long_window, min_periods=short_window).mean()
    out["trend_gap"] = out["close"] / out["trend_mean"].replace(0, np.nan) - 1

    # --- Drawdown from recent rolling high ---
    roll_max = out["close"].rolling(short_window, min_periods=1).max()
    out["drawdown"] = out["close"] / roll_max - 1

    # --- Consecutive down days ---
    neg = (out["return_1d"] < 0).astype(int)
    out["consec_down"] = neg.groupby((neg == 0).cumsum()).cumcount()

    # --- ATR as % of price ---
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(short_window, min_periods=short_window).mean()
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)

    feature_cols = list(CANONICAL_FEATURE_COLS)


    out = out.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        if col not in out.columns:
            raise ValueError(f"Expected engineered feature was not generated: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=feature_cols).reset_index(drop=True)

    # --- Downside-only forward label ---
    out["forward_return"] = out["close"].shift(-forward_horizon) / out["close"] - 1
    out["forward_min_return"] = (
        out["close"].shift(-1).rolling(forward_horizon).min().shift(-(forward_horizon - 1))
        / out["close"] - 1
    )
    out["forward_max_return"] = (
        out["close"].shift(-1).rolling(forward_horizon).max().shift(-(forward_horizon - 1))
        / out["close"] - 1
    )
    # forward_down_move: how far price DROPPED in the next N days (positive = worse)
    out["forward_down_move"] = (-out["forward_min_return"]).clip(lower=0)
    out["forward_abs_move"] = out["forward_down_move"]  # alias for compatibility

    labelled = out["forward_down_move"].notna()
    known_moves = out.loc[labelled, "forward_down_move"].dropna()

    requested_threshold = float(risk_threshold)
    effective_threshold = requested_threshold

    if len(known_moves) > 0:
        positives_at_requested = int((known_moves >= requested_threshold).sum())
        negatives_at_requested = int((known_moves < requested_threshold).sum())

        # Auto-relax if too few events — cap at 20% event rate (raised from 15%)
        if positives_at_requested < min_positive_events or negatives_at_requested == 0:
            effective_event_rate = min(0.20, max(min_positive_events / len(known_moves), 0.02))
            effective_threshold = float(known_moves.quantile(1 - effective_event_rate))

    out["requested_risk_threshold"] = requested_threshold
    out["effective_risk_threshold"] = effective_threshold

    out["future_risk_label"] = np.nan
    out.loc[labelled, "future_risk_label"] = (
        out.loc[labelled, "forward_down_move"] >= effective_threshold
    ).astype(int)

    return out, feature_cols


def explain_row(row: pd.Series, reference: pd.DataFrame, feature_cols: list[str], top_n: int | None = None) -> pd.DataFrame:
    ref = reference[feature_cols].apply(pd.to_numeric, errors="coerce")
    values = pd.to_numeric(row[feature_cols], errors="coerce")

    median = ref.median(numeric_only=True)
    mad = (ref - median).abs().median(numeric_only=True).replace(0, np.nan)

    deviation = ((values - median) / mad).abs()
    deviation = deviation.replace([np.inf, -np.inf], np.nan)

    rows = []
    for feature in feature_cols:
        raw_value = values.get(feature, np.nan)
        method = "MAD-scaled"
        score = deviation.get(feature, np.nan)

        if feature == "consec_down":
            series = ref[feature].dropna()
            if pd.isna(raw_value) or series.empty:
                score = 0.0
            else:
                percentile = float((series <= raw_value).mean())
                percentile = min(max(percentile, 0.0), 0.999999)
                # Convert empirical rarity to a positive score on a comparable
                # abnormality scale. This keeps zero-heavy count features visible
                # in the same chart without changing the model input itself.
                score = -np.log(max(1.0 - percentile, 1e-6))
            method = "Percentile-based"

        if pd.isna(score) or np.isinf(score):
            score = 0.0

        rows.append({
            "feature": str(feature),
            "feature_value": round(float(raw_value), 6) if not pd.isna(raw_value) else np.nan,
            "historical_median": round(float(median.get(feature, np.nan)), 6) if not pd.isna(median.get(feature, np.nan)) else np.nan,
            "deviation_score": round(float(score), 3),
            "deviation_method": method,
        })

    out = pd.DataFrame(rows)
    order_map = {name: i for i, name in enumerate(feature_cols)}
    out["_feature_order"] = out["feature"].map(order_map).fillna(len(order_map)).astype(int)
    out = out.sort_values(["deviation_score", "_feature_order"], ascending=[False, True], kind="mergesort").drop(columns=["_feature_order"])

    if top_n is not None:
        out = out.head(top_n)

    return out.reset_index(drop=True)
