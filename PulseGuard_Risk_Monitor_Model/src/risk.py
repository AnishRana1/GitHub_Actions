import numpy as np
from src.determinism import configure_determinism
configure_determinism()
import pandas as pd


LOW_RISK_EXPOSURE = 1.00
LOW_MEDIUM_RISK_EXPOSURE = 0.98
MEDIUM_RISK_EXPOSURE = 0.90
HIGH_RISK_EXPOSURE = 0.65
SEVERE_RISK_EXPOSURE = 0.10

# Legacy aliases
HIGH_MEDIUM_RISK_EXPOSURE = MEDIUM_RISK_EXPOSURE
CRASH_RISK_EXPOSURE = SEVERE_RISK_EXPOSURE


def safe_returns(series, lower: float = -0.95, upper: float = 0.95) -> pd.Series:
    """Convert returns to finite, bounded decimal returns for stable compounding."""
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=lower, upper=upper)


def safe_equity_curve(returns) -> pd.Series:
    """Build an equity curve without overflow from invalid or extreme returns."""
    r = safe_returns(pd.Series(returns))
    # log1p compounding is numerically more stable than repeated cumprod.
    log_curve = np.log1p(r).cumsum()
    log_curve = np.clip(log_curve, -50, 50)
    equity = np.exp(log_curve)
    return pd.Series(equity, index=r.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)


def classify_events(
    df: pd.DataFrame,
    probability_threshold: float = 0.50,
    low_medium_threshold: float | None = None,
    medium_threshold: float | None = None,
    high_threshold: float | None = None,
    severe_threshold: float | None = None,
    low_exposure: float = LOW_RISK_EXPOSURE,
    low_medium_exposure: float = LOW_MEDIUM_RISK_EXPOSURE,
    medium_exposure: float = MEDIUM_RISK_EXPOSURE,
    high_exposure: float = HIGH_RISK_EXPOSURE,
    severe_exposure: float = SEVERE_RISK_EXPOSURE,
    crash_exposure: float | None = None,
    high_medium_exposure: float | None = None,
) -> pd.DataFrame:
    """Classify ML crash probability into five risk bands.

    Band thresholds are passed explicitly from the app (each derived from its
    own signal-rate slider via threshold_from_signal_rate). If an explicit
    threshold is not provided, falls back to a fixed multiplier of
    probability_threshold for backward compatibility.

    Bands:
    - Low risk          prob < low_medium_threshold
    - Low-medium risk   prob >= low_medium_threshold
    - Medium risk       prob >= medium_threshold
    - High risk         prob >= high_threshold
    - Severe risk       prob >= severe_threshold   ← only this counts as crash warning
    """
    if crash_exposure is not None:
        severe_exposure = crash_exposure
    if high_medium_exposure is not None:
        medium_exposure = high_medium_exposure

    t = float(probability_threshold)
    lm_thr = low_medium_threshold if low_medium_threshold is not None else t * 0.30
    md_thr = medium_threshold     if medium_threshold     is not None else t * 0.55
    hi_thr = high_threshold       if high_threshold       is not None else t * 0.78
    sv_thr = severe_threshold     if severe_threshold     is not None else t

    out = df.copy()
    prob = out["ml_risk_probability"].fillna(0)

    out["risk_level"] = "Low risk"
    out.loc[prob >= lm_thr, "risk_level"] = "Low-medium risk"
    out.loc[prob >= md_thr, "risk_level"] = "Medium risk"
    out.loc[prob >= hi_thr, "risk_level"] = "High risk"
    out.loc[prob >= sv_thr, "risk_level"] = "Severe risk"

    out["ml_predicted_risk"] = out["risk_level"].eq("Severe risk")

    out["direction"] = "Flat"
    out.loc[out["return_1d"] > 0, "direction"] = "Up"
    out.loc[out["return_1d"] < 0, "direction"] = "Down"

    out["event_key"] = ""
    out.loc[out["risk_level"].eq("Low-medium risk"), "event_key"] = "LOW-MED"
    out.loc[out["risk_level"].eq("Medium risk"),     "event_key"] = "MED"
    out.loc[out["risk_level"].eq("High risk"),       "event_key"] = "HIGH"
    out.loc[out["risk_level"].eq("Severe risk"),     "event_key"] = "SEVERE"

    vol_component      = (out["volatility_ratio"].clip(0, 3).fillna(0) / 3)
    volume_component   = (out["volume_zscore"].abs().clip(0, 5).fillna(0) / 5)
    downside_component = (-out["return_1d"]).clip(0, 0.08).fillna(0) / 0.08

    out["risk_score"] = (
        0.65 * prob
        + 0.15 * vol_component
        + 0.10 * volume_component
        + 0.10 * downside_component
    ).clip(0, 1)

    out["recommendation"] = f"Low risk: use {low_exposure:.0%} exposure"
    out.loc[out["risk_level"].eq("Low-medium risk"), "recommendation"] = (
        f"Low-medium risk: use {low_medium_exposure:.0%} exposure"
    )
    out.loc[out["risk_level"].eq("Medium risk"), "recommendation"] = (
        f"Medium risk: use {medium_exposure:.0%} exposure"
    )
    out.loc[out["risk_level"].eq("High risk"), "recommendation"] = (
        f"High risk: use {high_exposure:.0%} exposure"
    )
    out.loc[out["risk_level"].eq("Severe risk"), "recommendation"] = (
        f"Severe risk: reduce to {severe_exposure:.0%} exposure"
    )
    return out


def rank_events(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    events = df[df["risk_level"].isin(
        ["Low-medium risk", "Medium risk", "High risk", "Severe risk"]
    )].copy()
    if events.empty:
        return events

    events["event_return_pct"] = events["return_1d"] * 100
    if "forward_down_move" in events.columns:
        events["forward_crash_move_pct"] = events["forward_down_move"] * 100
    else:
        events["forward_crash_move_pct"] = events["forward_abs_move"] * 100

    events["economic_significance"] = (
        events["event_return_pct"].clip(upper=0).abs().fillna(0)
        + 1.00 * events["forward_crash_move_pct"].fillna(0)
        + 0.50 * events["volume_zscore"].abs().fillna(0)
        + 4.00 * events["risk_score"].fillna(0)
    )

    cols = [
        "timestamp", "event_key", "risk_level", "direction", "recommendation",
        "close", "event_return_pct", "forward_crash_move_pct", "volume_zscore",
        "ml_risk_probability", "risk_score", "economic_significance",
    ]
    return events.sort_values(["economic_significance", "risk_score", "timestamp"], ascending=[False, False, True], kind="mergesort").head(top_n)[cols].copy()


def apply_risk_overlay(
    df: pd.DataFrame,
    cooldown_days: int = 1,
    low_exposure: float = LOW_RISK_EXPOSURE,
    low_medium_exposure: float = LOW_MEDIUM_RISK_EXPOSURE,
    medium_exposure: float = MEDIUM_RISK_EXPOSURE,
    high_exposure: float = HIGH_RISK_EXPOSURE,
    severe_exposure: float = SEVERE_RISK_EXPOSURE,
    crash_exposure: float | None = None,
    high_medium_exposure: float | None = None,
) -> pd.DataFrame:
    if crash_exposure is not None:
        severe_exposure = crash_exposure
    if high_medium_exposure is not None:
        medium_exposure = high_medium_exposure

    out = df.copy()
    desired = np.full(len(out), low_exposure, dtype=float)
    desired[out["risk_level"].eq("Low-medium risk").to_numpy()] = low_medium_exposure
    desired[out["risk_level"].eq("Medium risk").to_numpy()]     = medium_exposure
    desired[out["risk_level"].eq("High risk").to_numpy()]       = high_exposure
    desired[out["risk_level"].eq("Severe risk").to_numpy()]     = severe_exposure

    exposure = np.full(len(out), low_exposure, dtype=float)
    for i, exp in enumerate(desired):
        if exp < low_exposure:
            start = i + 1
            end = min(len(out), start + cooldown_days)
            exposure[start:end] = np.minimum(exposure[start:end], exp)

    out["risk_overlay_exposure"] = exposure
    return out


def backtest_overlay(df: pd.DataFrame, transaction_cost_bps: float = 0.0) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    # Use finite, bounded returns to prevent noisy Yahoo adjustments, bad ticks,
    # or divide-by-zero artefacts from exploding the equity curve.
    out["market_return"] = safe_returns(out["close"].pct_change())

    exposure = pd.to_numeric(out["risk_overlay_exposure"], errors="coerce").fillna(1.0).clip(0.0, 1.5)
    out["overlay_return_gross"] = safe_returns(exposure * out["market_return"])

    turnover = exposure.diff().abs().fillna(0).clip(0.0, 2.0)
    out["transaction_cost"] = (transaction_cost_bps / 10000.0) * turnover
    out["overlay_return"] = safe_returns(out["overlay_return_gross"] - out["transaction_cost"])

    out["equity_overlay"] = safe_equity_curve(out["overlay_return"]).to_numpy()
    out["equity_buy_hold"] = safe_equity_curve(out["market_return"]).to_numpy()

    daily = safe_returns(out["overlay_return"])
    vol = daily.std()
    sharpe = (daily.mean() / vol) * np.sqrt(252) if vol and not np.isclose(vol, 0) else 0.0

    overlay_peak = out["equity_overlay"].cummax().replace(0, np.nan)
    bh_peak = out["equity_buy_hold"].cummax().replace(0, np.nan)
    overlay_dd = (out["equity_overlay"] / overlay_peak - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    bh_dd = (out["equity_buy_hold"] / bh_peak - 1).replace([np.inf, -np.inf], np.nan).fillna(0)

    metrics = {
        "overlay_total_return":   float(out["equity_overlay"].iloc[-1] - 1),
        "buy_hold_total_return":  float(out["equity_buy_hold"].iloc[-1] - 1),
        "overlay_sharpe":         float(sharpe),
        "overlay_max_drawdown":   float(overlay_dd.min()),
        "buy_hold_max_drawdown":  float(bh_dd.min()),
        "average_exposure":       float(exposure.mean()),
        "risk_reduction_days":    int((exposure < 1.0).sum()),
    }
    return out, metrics
