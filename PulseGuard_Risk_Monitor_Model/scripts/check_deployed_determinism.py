"""Scheduled deterministic-output check for GitHub Actions.

Runs the same default modelling path twice in one clean process and fails if the
key outputs differ. This does not call or mutate the Streamlit app; it validates
that the deployed code and pinned environment are still repeatable.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.determinism import GLOBAL_RANDOM_SEED, configure_determinism, reset_random_seeds
from src.build_info import get_build_info
from src.data_loader import load_market_data, FROZEN_END_DATE
from src.features import make_features, FEATURE_SCHEMA_VERSION
from src.models import (
    predict_supervised_holdout,
    classification_metrics,
    rolling_isolation_forest_baseline,
    zscore_baseline,
)
from src.pipeline_cache import make_pipeline_cache_key
from src.risk import classify_events, apply_risk_overlay, backtest_overlay
from src.shared_scoring import compute_dashboard_style_baseline_returns

SNAPSHOT_PATH = PROJECT_ROOT / "deterministic_expected_output_snapshot.json"

DEFAULT_INPUTS = {
    "ticker": "AAPL",
    "period": "5y",
    "prediction_mode": "holdout",
    "window": 20,
    "forward_horizon": 5,
    "risk_threshold": 0.04,
    "probability_threshold": 0.50,
    "train_fraction": 0.70,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "max_features": 0.70,
    "n_estimators": 80,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "subsample": 0.80,
    "tcn_weight": 0.0,
    "initial_train_size": 300,
    "retrain_every": 60,
    "z_threshold": 3.0,
    "iforest_rate_pct": 5.0,
    "baseline_reduction": 0.50,
    "cooldown_days": 3,
}


def _normalise(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return round(value, 12)
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value




def compute_validation_stability(predicted, probability_threshold: float, n_windows: int = 4) -> float:
    """Return a 0-1 score for validation-window ROC-AUC consistency.

    This mirrors the Streamlit UI metric so the GitHub determinism snapshot also
    covers the displayed validation-stability value.
    """
    try:
        required = {"future_risk_label", "ml_risk_probability"}
        if predicted is None or not required.issubset(set(predicted.columns)):
            return float("nan")
        eval_df = predicted.copy()
        if "split" in eval_df.columns:
            eval_df = eval_df[eval_df["split"].ne("train")]
        eval_df = eval_df.dropna(subset=["future_risk_label", "ml_risk_probability"])
        if "timestamp" in eval_df.columns:
            eval_df = eval_df.sort_values("timestamp")
        if len(eval_df) < max(40, n_windows * 10):
            return float("nan")

        window_scores = []
        for window_positions in np.array_split(np.arange(len(eval_df)), n_windows):
            window = eval_df.iloc[window_positions]
            if len(window) < 10 or window["future_risk_label"].nunique() < 2:
                continue
            m = classification_metrics(
                window["future_risk_label"].astype(int),
                (window["ml_risk_probability"] >= float(probability_threshold)).astype(bool),
                window["ml_risk_probability"],
            )
            score = m.get("roc_auc", float("nan"))
            if score is not None and not np.isnan(float(score)):
                window_scores.append(float(score))

        if len(window_scores) < 2:
            return float("nan")
        mean_score = float(np.mean(window_scores))
        std_score = float(np.std(window_scores))
        if mean_score <= 0:
            return float("nan")
        stability = 1.0 - (std_score / mean_score)
        return float(np.clip(stability, 0.0, 1.0))
    except Exception:
        return float("nan")

def _digest(payload: dict) -> str:
    raw = json.dumps(_normalise(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_once() -> dict:
    reset_random_seeds(GLOBAL_RANDOM_SEED)
    market = load_market_data(DEFAULT_INPUTS["ticker"], DEFAULT_INPUTS["period"])
    featured, feature_cols = make_features(
        market.frame,
        short_window=DEFAULT_INPUTS["window"],
        forward_horizon=DEFAULT_INPUTS["forward_horizon"],
        risk_threshold=DEFAULT_INPUTS["risk_threshold"],
    )
    predicted, metrics = predict_supervised_holdout(
        featured,
        train_fraction=DEFAULT_INPUTS["train_fraction"],
        feature_cols=feature_cols,
        probability_threshold=DEFAULT_INPUTS["probability_threshold"],
        max_depth=DEFAULT_INPUTS["max_depth"],
        min_samples_leaf=DEFAULT_INPUTS["min_samples_leaf"],
        max_features=DEFAULT_INPUTS["max_features"],
        n_estimators=DEFAULT_INPUTS["n_estimators"],
        learning_rate=DEFAULT_INPUTS["learning_rate"],
        reg_lambda=DEFAULT_INPUTS["reg_lambda"],
        reg_alpha=DEFAULT_INPUTS["reg_alpha"],
        subsample=DEFAULT_INPUTS["subsample"],
        use_tcn=False,
        tcn_weight=DEFAULT_INPUTS["tcn_weight"],
        train_tcn_if_missing=False,
    )
    predicted = rolling_isolation_forest_baseline(
        predicted,
        feature_cols=feature_cols,
        initial_train_size=min(DEFAULT_INPUTS["initial_train_size"], max(60, len(predicted) // 2)),
        contamination=DEFAULT_INPUTS["iforest_rate_pct"] / 100.0,
        retrain_every=DEFAULT_INPUTS["retrain_every"],
    )
    predicted = zscore_baseline(predicted, threshold=DEFAULT_INPUTS["z_threshold"])

    classified = classify_events(predicted)
    overlaid = apply_risk_overlay(classified)
    backtested, overlay_metrics = backtest_overlay(overlaid)

    baseline_metrics = compute_dashboard_style_baseline_returns(
        backtested,
        cooldown_days=DEFAULT_INPUTS["cooldown_days"],
        baseline_reduction=DEFAULT_INPUTS["baseline_reduction"],
    )

    prob_series = backtested["ml_risk_probability"].dropna().round(12).tolist()
    risk_counts = backtested["risk_level"].astype(str).value_counts().sort_index().to_dict() if "risk_level" in backtested else {}
    valid_prob = backtested["ml_risk_probability"].notna() if "ml_risk_probability" in backtested else np.zeros(len(backtested), dtype=bool)
    severe_warning_rate = float(backtested.loc[valid_prob, "ml_predicted_risk"].fillna(False).mean()) if valid_prob.any() else float("nan")
    validation_stability = compute_validation_stability(predicted, DEFAULT_INPUTS["probability_threshold"])

    dashboard_metrics = {
        "roc_auc": metrics.get("roc_auc", float("nan")),
        "pr_auc": metrics.get("pr_auc", float("nan")),
        "recall": metrics.get("recall", float("nan")),
        "precision": metrics.get("precision", float("nan")),
        "sharpe_ratio": overlay_metrics.get("overlay_sharpe", float("nan")),
        "severe_warning_rate": severe_warning_rate,
        "peak_drawdown": overlay_metrics.get("overlay_max_drawdown", float("nan")),
        "validation_stability": validation_stability,
        "ml_overlay_return": overlay_metrics.get("overlay_total_return", float("nan")),
        "buy_hold_return": overlay_metrics.get("buy_hold_total_return", float("nan")),
        "zscore_baseline_return": baseline_metrics.get("zscore_return", float("nan")),
        "isolation_forest_return": baseline_metrics.get("iforest_return", float("nan")),
    }

    cache_inputs = {k: v for k, v in DEFAULT_INPUTS.items() if k not in {"initial_train_size", "retrain_every", "z_threshold", "iforest_rate_pct", "baseline_reduction", "cooldown_days"}}
    cache_key = make_pipeline_cache_key(**cache_inputs)
    return {
        "seed": GLOBAL_RANDOM_SEED,
        "frozen_data_cutoff": str(FROZEN_END_DATE.date()),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "shape": list(featured.shape),
        "feature_cols": list(feature_cols),
        "probability_digest": _digest({"probs": prob_series}),
        "risk_counts": risk_counts,
        "metrics": metrics,
        "validation_stability": validation_stability,
        "overlay_metrics": overlay_metrics,
        "dashboard_metrics": dashboard_metrics,
    }


def _load_expected_snapshot() -> dict | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except Exception as exc:
        raise SystemExit(f"Could not read deterministic snapshot: {exc}")


def main() -> None:
    configure_determinism()
    first = run_once()
    second = run_once()
    if _digest(first) != _digest(second):
        print("Determinism check failed: repeated runs produced different outputs.")
        print(json.dumps({"first": first, "second": second}, indent=2, sort_keys=True, default=str))
        raise SystemExit(1)

    expected = _load_expected_snapshot()
    if expected is not None:
        expected_output = expected.get("default_output")
        current_output = _normalise(first)
        if _digest(expected_output) != _digest(current_output):
            print("Determinism snapshot check failed: current output differs from approved snapshot.")
            print(json.dumps({"expected": expected_output, "current": current_output}, indent=2, sort_keys=True, default=str))
            raise SystemExit(1)

        expected_build = expected.get("build_info", {})
        current_build = get_build_info()
        if expected_build.get("build_id") != current_build.get("build_id"):
            print("Build traceability warning: build ID differs from the snapshot build ID.")
            print(json.dumps({"snapshot_build": expected_build, "current_build": current_build}, indent=2, sort_keys=True, default=str))

    print("Determinism check passed.")
    print(json.dumps(first, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
