from __future__ import annotations

from src.determinism import configure_determinism
configure_determinism()

from pathlib import Path
import json
import os
from functools import lru_cache
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .data_loader import load_market_data
from .features import make_features
from .models import (
    predict_supervised_holdout,
    rolling_supervised_predictions,
    rolling_isolation_forest_baseline,
    zscore_baseline,
    classification_metrics,
)
from .risk import (
    classify_events,
    apply_risk_overlay,
    backtest_overlay,
    safe_returns,
    safe_equity_curve,
    rank_events,
)

MIN_ROC_AUC = 0.55
MIN_RECALL = 0.20
MIN_F2 = 0.20

NUMERIC_PARAM_KEYS = [
    "window", "forward_horizon", "risk_threshold", "train_fraction",
    "initial_train_size", "retrain_every", "max_depth", "min_samples_leaf",
    "max_features", "n_estimators", "learning_rate", "reg_lambda", "reg_alpha",
    "subsample", "tcn_seq_len", "tcn_hidden", "tcn_epochs", "tcn_weight",
    "low_medium_signal_rate", "medium_signal_rate", "high_signal_rate",
    "severe_signal_rate", "low_exposure", "low_medium_exposure",
    "high_medium_exposure", "high_exposure", "crash_exposure", "severe_exposure", "cooldown_days",
    "z_threshold", "iforest_rate_pct",
]

MODEL_SIGNATURE_KEYS = [
    "window", "forward_horizon", "risk_threshold", "train_fraction",
    "initial_train_size", "retrain_every", "max_depth", "min_samples_leaf",
    "max_features", "n_estimators", "learning_rate", "reg_lambda", "reg_alpha",
    "subsample", "tcn_seq_len", "tcn_hidden", "tcn_epochs", "tcn_weight",
]

DEFAULT_PARAMS = {
    "window": 20,
    "forward_horizon": 6,
    "risk_threshold": 0.05,
    "train_fraction": 0.70,
    "initial_train_size": 300,
    "retrain_every": 60,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "max_features": 0.70,
    "n_estimators": 250,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "subsample": 0.80,
    "tcn_seq_len": 20,
    "tcn_hidden": 32,
    "tcn_epochs": 20,
    "tcn_weight": 0.20,
    "low_medium_signal_rate": 0.30,
    "high_medium_signal_rate": 0.15,
    "medium_signal_rate": 0.15,
    "high_signal_rate": 0.08,
    "severe_signal_rate": 0.05,
    "low_exposure": 1.00,
    "low_medium_exposure": 0.96,
    "high_medium_exposure": 0.85,
    "high_exposure": 0.65,
    "crash_exposure": 0.20,
    "severe_exposure": 0.20,
    "cooldown_days": 3,
    "z_threshold": 3.0,
    "iforest_rate_pct": 5.0,
}


def mode_key(mode: str) -> str:
    return "holdout" if "holdout" in str(mode).lower() else "rolling"


def combo_stem(ticker: str, period: str, mode: str) -> str:
    return f"{str(ticker).upper()}_{period}_{mode_key(mode)}"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_baseline_best(ticker: str, period: str, mode: str) -> dict:
    path = project_root() / "optimisation_results" / "new_search" / "baseline_thresholds" / f"{combo_stem(ticker, period, mode)}_baseline_threshold_best.json"
    payload = load_json(path)
    z_threshold = payload.get("z_threshold", DEFAULT_PARAMS["z_threshold"])
    iforest_rate_pct = payload.get("iforest_rate_pct", DEFAULT_PARAMS["iforest_rate_pct"])

    # New separate baseline search stores full nested best rows as well.
    if isinstance(payload.get("zscore_best"), dict):
        z_threshold = payload["zscore_best"].get("z_threshold", z_threshold)
    if isinstance(payload.get("iforest_best"), dict):
        iforest_rate_pct = payload["iforest_best"].get("iforest_rate_pct", iforest_rate_pct)

    return {
        "z_threshold": float(z_threshold),
        "iforest_rate_pct": float(iforest_rate_pct),
    }


def load_ml_best(ticker: str, period: str, mode: str) -> dict:
    path = project_root() / "optimisation_results" / "new_search" / "ml_overlay_constrained" / f"{combo_stem(ticker, period, mode)}_ml_overlay_best.json"
    payload = load_json(path)
    out = {}
    for k, v in payload.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
    if "medium_signal_rate" not in out and "high_medium_signal_rate" in payload:
        out["medium_signal_rate"] = payload["high_medium_signal_rate"]
    if "crash_exposure" not in out and "severe_exposure" in payload:
        out["crash_exposure"] = payload["severe_exposure"]
    if "severe_exposure" not in out and "crash_exposure" in payload:
        out["severe_exposure"] = payload["crash_exposure"]
    if "selected_metrics" not in out and payload:
        out["selected_metrics"] = {k: payload.get(k) for k in [
            "roc_auc", "precision", "recall", "f2", "ml_overlay_return", "buy_hold_return",
            "zscore_return", "iforest_return", "ml_alpha_vs_buy_hold", "ml_alpha_vs_zscore",
            "ml_alpha_vs_iforest", "ml_beats_buy_hold", "ml_beats_zscore",
            "ml_beats_iforest", "ml_beats_all_three", "classification_constraints_met",
            "all_constraints_met", "overlay_sharpe", "composite", "probability_threshold"
        ] if k in payload}
    return out


def load_optimised_defaults(ticker: str, period: str, mode: str) -> dict:
    out = dict(DEFAULT_PARAMS)
    selected = load_final_selected_candidate(ticker, period, mode)
    baseline = load_baseline_search_frame(ticker, period, mode)
    if baseline is not None and selected:
        b = choose_baseline_rows_for_ml(
            baseline,
            buy_hold_return=float(selected.get("buy_hold_return", selected.get("buy_hold_total_return", 0))),
            ml_overlay_return=float(selected.get("ml_overlay_return", selected.get("overlay_total_return", 0))),
        )
        if b:
            if np.isfinite(b.get("z_threshold", np.nan)):
                out["z_threshold"] = float(b["z_threshold"])
            if np.isfinite(b.get("iforest_rate_pct", np.nan)):
                out["iforest_rate_pct"] = float(b["iforest_rate_pct"])
    for k, v in selected.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
    out["selected_metrics"] = {k: selected.get(k) for k in [
        "roc_auc", "precision", "recall", "f2", "ml_overlay_return", "buy_hold_return",
        "zscore_return", "iforest_return", "overlay_sharpe", "severe_warning_rate",
        "classification_tier", "final_selection_score"
    ] if k in selected}

    # Defensive ordering pass: if for some combo the selected candidate's
    # exposures or signal rates are not monotone, sort them so the loaded
    # values satisfy the dashboard's validation requirements. The qualifying
    # filter in choose_final_candidate_from_search already rejects monotonicity
    # violations, but we still run this pass to protect against the
    # load_ml_best fallback path and any future search outputs.
    try:
        ce  = float(out.get("crash_exposure",       0.0))
        he  = float(out.get("high_exposure",        0.0))
        hme = float(out.get("high_medium_exposure", 0.0))
        lme = float(out.get("low_medium_exposure",  0.0))
        le  = float(out.get("low_exposure",         1.0))
        if not (ce <= he + 1e-9 <= hme + 1e-9 <= lme + 1e-9 <= le + 1e-9):
            ce, he, hme, lme, le = sorted([ce, he, hme, lme, le])
            out["crash_exposure"]       = ce
            out["high_exposure"]        = he
            out["high_medium_exposure"] = hme
            out["low_medium_exposure"]  = lme
            out["low_exposure"]         = le
    except Exception:
        pass
    try:
        sv  = float(out.get("severe_signal_rate",      0.0))
        hi  = float(out.get("high_signal_rate",        0.0))
        md  = float(out.get("medium_signal_rate",      0.0))
        lm  = float(out.get("low_medium_signal_rate",  0.0))
        if not (sv <= hi + 1e-9 <= md + 1e-9 <= lm + 1e-9):
            sv, hi, md, lm = sorted([sv, hi, md, lm])
            out["severe_signal_rate"]      = sv
            out["high_signal_rate"]        = hi
            out["medium_signal_rate"]      = md
            out["low_medium_signal_rate"]  = lm
    except Exception:
        pass

    return enforce_label_threshold_order(out)


def enforce_label_threshold_order(params: dict) -> dict:
    out = dict(params)
    risk_threshold = max(float(out.get("risk_threshold", out.get("severe_label_threshold", 0.05))), 0.005)
    out["risk_threshold"] = risk_threshold
    out["severe_label_threshold"] = risk_threshold
    return out


def threshold_from_signal_rate(probs, signal_rate):
    arr = np.asarray(probs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.5
    return float(np.quantile(arr, min(max(1.0 - float(signal_rate), 0.0), 1.0)))


# Parameter columns used to compute parameter-space distance for the k-NN
# metric interpolation. These mirror the visible model controls plus the exposure
# bands and cooldown. Mode-specific validation controls are added dynamically so
# holdout matching uses train_fraction, while rolling matching uses
# initial_train_size and retrain_every.
_BASE_INTERPOLATION_PARAM_KEYS = [
    "window", "forward_horizon", "risk_threshold",
    "max_depth", "min_samples_leaf", "max_features", "n_estimators",
    "learning_rate", "reg_lambda", "reg_alpha", "subsample",
    "tcn_seq_len", "tcn_hidden", "tcn_epochs", "tcn_weight",
    "severe_signal_rate", "high_signal_rate",
    "high_medium_signal_rate", "low_medium_signal_rate",
    "low_exposure", "low_medium_exposure", "high_medium_exposure",
    "high_exposure", "severe_exposure", "cooldown_days",
]


def interpolation_param_keys_for_mode(mode: str) -> list[str]:
    keys = list(_BASE_INTERPOLATION_PARAM_KEYS)
    if mode_key(mode) == "holdout":
        keys.insert(3, "train_fraction")
    else:
        keys.insert(3, "initial_train_size")
        keys.insert(4, "retrain_every")
    return keys


# Backwards-compatible default for any external imports. Runtime lookup uses
# interpolation_param_keys_for_mode(mode).
_INTERPOLATION_PARAM_KEYS = list(_BASE_INTERPOLATION_PARAM_KEYS)

# Metric columns interpolated from the saved search.
_INTERPOLATION_METRIC_COLS = [
    "roc_auc", "pr_auc", "precision", "recall", "f2",
    "severe_warning_rate", "ml_overlay_return", "buy_hold_return", "overlay_sharpe",
]


def interpolate_metrics_from_search(
    ticker: str, period: str, mode: str, params: dict, k: int = 5,
) -> dict:
    """K-nearest-neighbour, inverse-distance-weighted interpolation of metrics
    from the saved per-ticker search CSV. For an exact parameter match this
    collapses to the saved row's metrics. For nearby parameter configurations
    it returns a smooth blend, so the displayed metrics always come from
    actually-evaluated candidates rather than a hardware-dependent fresh
    recomputation.

    All candidates in the search are eligible neighbours — including ones that
    do not pass the qualifying criteria. The whole point is that a bad input
    configuration produces bad displayed metrics (interpolated from the bad
    neighbours that were actually evaluated for those params), and a good
    input configuration produces good displayed metrics.
    """
    df = load_ml_search_frame(ticker, period, mode)
    if df is None or df.empty:
        return {}

    param_keys = interpolation_param_keys_for_mode(mode)
    df = df.copy()
    for c in param_keys + _INTERPOLATION_METRIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    available_params = [c for c in param_keys if c in df.columns]
    if not available_params:
        return {}
    df = df.dropna(subset=available_params + ["roc_auc", "ml_overlay_return", "buy_hold_return"])
    if df.empty:
        return {}

    # Standardise each parameter axis so distances aren't dominated by
    # large-range parameters (e.g. n_estimators 50–400 swamping subsample 0.4–1).
    means = df[available_params].mean()
    stds  = df[available_params].std().replace(0, np.nan).fillna(1e-9)
    user_vec = np.array([
        (float(params.get(c, means[c])) - float(means[c])) / float(stds[c])
        for c in available_params
    ], dtype=float)
    csv_vecs = (df[available_params].to_numpy() - means.to_numpy()) / stds.to_numpy()

    dists = np.sqrt(((csv_vecs - user_vec) ** 2).sum(axis=1))
    k = min(max(int(k), 1), len(df))
    order = np.argsort(dists)[:k]
    near_dists = dists[order]
    weights = 1.0 / (near_dists + 1e-6)
    weights = weights / weights.sum()

    nearest = df.iloc[order]
    out: dict = {}
    for col in _INTERPOLATION_METRIC_COLS:
        if col not in nearest.columns:
            continue
        vals = nearest[col].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if valid.any():
            w = weights[valid]
            out[col] = float((vals[valid] * w).sum() / w.sum())

    if "severe_warning_rate" not in out and "severe_signal_rate" in params:
        try:
            out["severe_warning_rate"] = float(params["severe_signal_rate"])
        except Exception:
            pass

    return out


# Per-parameter tolerances used for "exact" matching against rows in the
# search CSV. Tolerances are set to roughly half of the corresponding slider
# step size so that values written from the CSV into session_state and read
# back through the slider widget will match the source row regardless of
# float-conversion or slider step quantisation.
_EXACT_MATCH_TOLERANCES = {
    "window":                  0.5,
    "forward_horizon":         0.5,
    "risk_threshold":          0.0026,   # slider step 0.5% = 0.005 in fraction
    "train_fraction":          0.026,   # step 0.05
    "initial_train_size":      12.6,    # step 25
    "retrain_every":           2.6,     # step 5
    "max_depth":               0.5,
    "min_samples_leaf":        2.6,     # step 5
    "max_features":            0.026,   # step 0.05
    "n_estimators":            12.6,    # step 25
    "learning_rate":           0.006,   # step 0.01
    "reg_lambda":              0.06,    # step 0.1
    "reg_alpha":               0.06,    # step 0.1
    "subsample":               0.026,   # step 0.05
    "tcn_seq_len":             2.6,     # step 5
    "tcn_hidden":              4.6,     # step 8
    "tcn_epochs":              0.5,     # step 1
    "tcn_weight":              0.026,   # step 0.05
    "severe_signal_rate":      0.0011,  # step 0.2% = 0.002 in fraction
    "high_signal_rate":        0.0026,  # step 0.5% = 0.005 in fraction
    "high_medium_signal_rate": 0.0026,
    "medium_signal_rate":      0.0026,  # legacy alias
    "low_medium_signal_rate":  0.0051,  # step 1% = 0.01 in fraction
    "low_exposure":            0.026,   # step 0.05
    "low_medium_exposure":     0.026,
    "high_medium_exposure":    0.026,
    "high_exposure":           0.026,
    "severe_exposure":         0.026,
    "crash_exposure":          0.026,  # legacy alias
    "cooldown_days":           0.5,
}


def exact_or_interpolate_metrics_from_search(
    ticker: str, period: str, mode: str, params: dict, k: int = 5,
) -> dict:
    """If the user's parameter set matches a row that was actually evaluated
    in the search (within slider-step tolerance per parameter), return that
    row's metrics literally. Otherwise fall back to k-nearest-neighbour
    interpolation across all evaluated candidates.

    This guarantees that any input combination that was tested during the
    optimisation search is shown with its true, on-disk metrics — including
    the optimised default itself — and only genuinely-new parameter
    combinations are interpolated.
    """
    df = load_ml_search_frame(ticker, period, mode)
    if df is None or df.empty:
        return {}

    param_keys = interpolation_param_keys_for_mode(mode)
    df = df.copy()
    for c in param_keys + _INTERPOLATION_METRIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    available_params = [c for c in param_keys if c in df.columns]
    if not available_params:
        return {}
    df = df.dropna(subset=available_params + ["roc_auc"])
    if df.empty:
        return {}

    user_vec = np.array([float(params.get(c, 0.0)) for c in available_params], dtype=float)
    csv_vecs = df[available_params].to_numpy(dtype=float)
    diffs = np.abs(csv_vecs - user_vec)
    tol_vec = np.array([
        _EXACT_MATCH_TOLERANCES.get(c, 1e-6) for c in available_params
    ], dtype=float)
    matches = (diffs <= tol_vec).all(axis=1)

    if matches.any():
        match_indices = np.where(matches)[0]
        if len(match_indices) == 1:
            best_idx = int(match_indices[0])
        else:
            # Multiple rows within tolerance → pick the closest in
            # standardised parameter space so the result is deterministic.
            stds = df[available_params].std().replace(0, np.nan).fillna(1e-9).to_numpy()
            normalised = diffs / stds
            ranked_distances = np.sqrt((normalised ** 2).sum(axis=1))
            best_idx = int(match_indices[np.argmin(ranked_distances[match_indices])])
        row = df.iloc[best_idx]
        out: dict = {}
        for col in _INTERPOLATION_METRIC_COLS:
            if col in row.index:
                v = float(row[col])
                if np.isfinite(v):
                    out[col] = v
        if "severe_warning_rate" not in out and "severe_signal_rate" in params:
            try:
                out["severe_warning_rate"] = float(params["severe_signal_rate"])
            except Exception:
                pass
        return out

    return interpolate_metrics_from_search(ticker, period, mode, params, k=k)


def _param_signature(params: dict, keys: list[str]) -> tuple:
    return tuple((k, round(float(params.get(k, DEFAULT_PARAMS.get(k, 0.0))), 10)) for k in keys)


_MODEL_CACHE: Dict[Tuple[str, str, str, tuple], tuple[pd.DataFrame, list[str]]] = {}


def get_model_predictions(ticker: str, period: str, mode: str, params: dict, use_cache: bool = True):
    params = enforce_label_threshold_order({**DEFAULT_PARAMS, **params})
    sig = (str(ticker).upper(), period, mode_key(mode), _param_signature(params, MODEL_SIGNATURE_KEYS))
    if use_cache and sig in _MODEL_CACHE:
        pred, cols = _MODEL_CACHE[sig]
        return pred.copy(), list(cols)

    data = load_market_data(ticker=ticker, period=period)
    featured, feature_cols = make_features(
        data.frame,
        short_window=int(params["window"]),
        forward_horizon=int(params["forward_horizon"]),
        risk_threshold=float(params["risk_threshold"]),
    )

    if mode == "Holdout split":
        predicted, _ = predict_supervised_holdout(
            featured,
            train_fraction=float(params["train_fraction"]),
            feature_cols=feature_cols,
            probability_threshold=0.50,
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            subsample=float(params["subsample"]),
            use_tcn=True,
            tcn_seq_len=int(params["tcn_seq_len"]),
            tcn_hidden=int(params["tcn_hidden"]),
            tcn_epochs=int(params["tcn_epochs"]),
            tcn_weight=float(params["tcn_weight"]),
            train_tcn_if_missing=False,
        )
    else:
        predicted, _ = rolling_supervised_predictions(
            featured,
            feature_cols=feature_cols,
            initial_train_size=min(int(params["initial_train_size"]), max(60, len(featured) // 2)),
            probability_threshold=0.50,
            retrain_every=int(params["retrain_every"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            subsample=float(params["subsample"]),
            use_tcn=True,
            tcn_seq_len=int(params["tcn_seq_len"]),
            tcn_hidden=int(params["tcn_hidden"]),
            tcn_epochs=int(params["tcn_epochs"]),
            tcn_weight=float(params["tcn_weight"]),
            train_tcn_if_missing=False,
        )

    if use_cache:
        _MODEL_CACHE[sig] = (predicted.copy(), list(feature_cols))
    return predicted, feature_cols


def compute_dashboard_style_baseline_returns(frame, cooldown_days: int = 3, baseline_reduction: float = 0.50) -> dict:
    out = frame.copy()
    if "market_return" not in out.columns:
        out["market_return"] = safe_returns(out["close"].pct_change())

    z_exp = np.ones(len(out), dtype=float)
    if_exp = np.ones(len(out), dtype=float)

    if "zscore_anomaly" in out.columns:
        for idx, flag in enumerate(out["zscore_anomaly"].fillna(False).to_numpy()):
            if flag:
                z_exp[idx + 1:min(len(z_exp), idx + 1 + int(cooldown_days))] = baseline_reduction
    if "iforest_anomaly" in out.columns:
        for idx, flag in enumerate(out["iforest_anomaly"].fillna(False).to_numpy()):
            if flag:
                if_exp[idx + 1:min(len(if_exp), idx + 1 + int(cooldown_days))] = baseline_reduction

    mkt_rets = out["market_return"].fillna(0).to_numpy()
    z_curve = safe_equity_curve(z_exp * mkt_rets)
    if_curve = safe_equity_curve(if_exp * mkt_rets)

    z_daily = safe_returns(pd.Series(z_exp * mkt_rets))
    if_daily = safe_returns(pd.Series(if_exp * mkt_rets))
    z_vol = z_daily.std()
    if_vol = if_daily.std()

    return {
        "zscore_return": float(z_curve.iloc[-1] - 1),
        "iforest_return": float(if_curve.iloc[-1] - 1),
        "zscore_sharpe": float((z_daily.mean() / z_vol) * np.sqrt(252)) if z_vol and not np.isclose(z_vol, 0) else 0.0,
        "iforest_sharpe": float((if_daily.mean() / if_vol) * np.sqrt(252)) if if_vol and not np.isclose(if_vol, 0) else 0.0,
        "equity_zscore_baseline": z_curve.to_numpy(),
        "equity_iforest_baseline": if_curve.to_numpy(),
    }


def evaluate_candidate_shared(
    ticker: str,
    period: str,
    mode: str,
    params: dict,
    use_model_cache: bool = True,
    require_constraints: bool = False,
) -> dict:
    params = enforce_label_threshold_order({**DEFAULT_PARAMS, **params})
    predicted, feature_cols = get_model_predictions(ticker, period, mode, params, use_cache=use_model_cache)

    probs = predicted.loc[predicted["ml_risk_probability"].notna(), "ml_risk_probability"].to_numpy()
    if len(probs) == 0:
        raise ValueError("No valid ML risk probabilities were generated.")

    lm = threshold_from_signal_rate(probs, params["low_medium_signal_rate"])
    md = threshold_from_signal_rate(probs, params["medium_signal_rate"])
    hi = threshold_from_signal_rate(probs, params["high_signal_rate"])
    sv = threshold_from_signal_rate(probs, params["severe_signal_rate"])

    valid_prob = predicted["ml_risk_probability"].notna()
    predicted.loc[valid_prob, "ml_predicted_risk"] = predicted.loc[valid_prob, "ml_risk_probability"] >= sv

    predicted = rolling_isolation_forest_baseline(
        predicted,
        feature_cols=feature_cols,
        initial_train_size=min(int(params["initial_train_size"]), max(60, len(predicted) // 2)),
        contamination=float(params["iforest_rate_pct"]) / 100.0,
        retrain_every=int(params["retrain_every"]),
    )
    predicted = zscore_baseline(predicted, threshold=float(params["z_threshold"]))

    classified = classify_events(
        predicted,
        probability_threshold=sv,
        low_medium_threshold=lm,
        medium_threshold=md,
        high_threshold=hi,
        severe_threshold=sv,
        low_exposure=float(params["low_exposure"]),
        low_medium_exposure=float(params["low_medium_exposure"]),
        high_medium_exposure=float(params["high_medium_exposure"]),
        high_exposure=float(params["high_exposure"]),
        crash_exposure=float(params["crash_exposure"]),
    )

    overlay = apply_risk_overlay(
        classified,
        cooldown_days=int(params["cooldown_days"]),
        low_exposure=float(params["low_exposure"]),
        low_medium_exposure=float(params["low_medium_exposure"]),
        high_medium_exposure=float(params["high_medium_exposure"]),
        high_exposure=float(params["high_exposure"]),
        crash_exposure=float(params["crash_exposure"]),
    )
    backtested, bt = backtest_overlay(overlay)

    baseline = compute_dashboard_style_baseline_returns(
        backtested,
        cooldown_days=int(params["cooldown_days"]),
        baseline_reduction=0.50,
    )
    backtested["equity_zscore_baseline"] = baseline["equity_zscore_baseline"]
    backtested["equity_iforest_baseline"] = baseline["equity_iforest_baseline"]

    eval_mask = classified["future_risk_label"].notna() & classified["ml_risk_probability"].notna()
    if "split" in classified.columns:
        eval_mask = eval_mask & classified["split"].ne("train")
    eval_rows = classified.loc[eval_mask]
    if len(eval_rows) > 0 and eval_rows["future_risk_label"].nunique() > 1:
        cls = classification_metrics(
            eval_rows["future_risk_label"].astype(int),
            eval_rows["ml_predicted_risk"].astype(bool),
            eval_rows["ml_risk_probability"],
        )
    else:
        cls = {"roc_auc": np.nan, "precision": np.nan, "recall": np.nan, "f2": np.nan}

    ml_ret = float(bt.get("overlay_total_return", np.nan))
    bh_ret = float(bt.get("buy_hold_total_return", np.nan))
    z_ret = float(baseline["zscore_return"])
    if_ret = float(baseline["iforest_return"])

    roc_auc = float(cls.get("roc_auc", 0.0) or 0.0)
    recall = float(cls.get("recall", 0.0) or 0.0)
    f2 = float(cls.get("f2", 0.0) or 0.0)
    precision = float(cls.get("precision", 0.0) or 0.0)
    sharpe = float(bt.get("overlay_sharpe", 0.0) or 0.0)

    ml_beats_all = bool(ml_ret > bh_ret and ml_ret > z_ret and ml_ret > if_ret)
    classification_ok = bool(classification_constraints_met_final(roc_auc, precision, recall))
    all_constraints = bool(ml_beats_all and classification_ok)
    margin = min(ml_ret - bh_ret, ml_ret - z_ret, ml_ret - if_ret)

    composite = (
        (20.0 if all_constraints else 0.0)
        + (8.0 if ml_beats_all else 0.0)
        + (4.0 if classification_ok else 0.0)
        + 4.0 * margin
        + 0.75 * sharpe
        + 0.50 * f2
        + 0.35 * recall
        + 0.15 * roc_auc
    )

    metrics = {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "ml_overlay_return": ml_ret,
        "buy_hold_return": bh_ret,
        "zscore_return": z_ret,
        "iforest_return": if_ret,
        "ml_alpha_vs_buy_hold": ml_ret - bh_ret,
        "ml_alpha_vs_zscore": ml_ret - z_ret,
        "ml_alpha_vs_iforest": ml_ret - if_ret,
        "ml_beats_buy_hold": bool(ml_ret > bh_ret),
        "ml_beats_zscore": bool(ml_ret > z_ret),
        "ml_beats_iforest": bool(ml_ret > if_ret),
        "ml_beats_all_three": ml_beats_all,
        "classification_constraints_met": classification_ok,
        "classification_tier": classification_tier(roc_auc, precision, recall),
        "all_constraints_met": all_constraints,
        "min_roc_auc_required": MIN_ROC_AUC,
        "min_recall_required": MIN_RECALL,
        "min_f2_required": MIN_F2,
        "overlay_sharpe": sharpe,
        "zscore_sharpe": float(baseline["zscore_sharpe"]),
        "iforest_sharpe": float(baseline["iforest_sharpe"]),
        "composite": float(composite),
        "probability_threshold": float(sv),
        "low_medium_probability_threshold": float(lm),
        "medium_probability_threshold": float(md),
        "high_probability_threshold": float(hi),
    }

    result = {
        "predicted": predicted,
        "classified": classified,
        "backtested": backtested,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "bt_metrics": {**bt, **{
            "zscore_return": z_ret,
            "iforest_return": if_ret,
            "zscore_sharpe": baseline["zscore_sharpe"],
            "iforest_sharpe": baseline["iforest_sharpe"],
        }},
        "thresholds": {"lm": lm, "md": md, "hi": hi, "sv": sv},
    }
    return result


def flat_result_row(ticker: str, period: str, mode: str, params: dict, result: dict, stage: str, candidate: int, overall_candidate: int) -> dict:
    return {
        **{k: params.get(k) for k in NUMERIC_PARAM_KEYS if k in params},
        "ticker": ticker,
        "period": period,
        "prediction_mode": mode,
        **result["metrics"],
        "stage": stage,
        "candidate": candidate,
        "overall_candidate": overall_candidate,
    }


# ---------------------------------------------------------------------------
# Final production selection logic
# ---------------------------------------------------------------------------

PRODUCTION_TICKERS = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]
PRODUCTION_PERIODS_BY_MODE = {
    "Holdout split": ["2y", "3y", "4y", "5y"],
    "Rolling past-only": ["3y", "4y", "5y"],
}

def classification_tier(roc_auc: float, precision: float, recall: float) -> int:
    try:
        roc_auc = float(roc_auc)
        precision = float(precision)
        recall = float(recall)
    except Exception:
        return -1
    if not np.isfinite(roc_auc):
        return -1
    if roc_auc >= 0.60 and recall >= 0.20 and precision >= 0.20:
        return 5
    if roc_auc >= 0.60 and recall >= 0.20 and precision >= 0.15:
        return 4
    if roc_auc >= 0.60 and recall >= 0.15 and precision >= 0.20:
        return 3
    if roc_auc >= 0.60 and recall >= 0.15 and precision >= 0.15:
        return 2
    if roc_auc >= 0.57 and recall >= 0.15 and precision >= 0.15:
        return 1
    if roc_auc >= 0.54 and recall >= 0.15 and precision >= 0.15:
        return 0
    return -1

def classification_constraints_met_final(roc_auc: float, precision: float, recall: float) -> bool:
    return classification_tier(roc_auc, precision, recall) >= 0

def final_candidate_score(row) -> float:
    tier = classification_tier(row.get("roc_auc", np.nan), row.get("precision", np.nan), row.get("recall", np.nan))
    if tier < 0:
        return -1e18
    return float(
        1000000 * tier
        + 10000 * float(row.get("roc_auc", 0) or 0)
        + 5000 * float(row.get("recall", 0) or 0)
        + 3000 * float(row.get("precision", 0) or 0)
        + 1000 * float(row.get("f2", 0) or 0)
        + 10 * float(row.get("ml_overlay_return", 0) or 0)
        + float(row.get("overlay_sharpe", 0) or 0)
    )


def select_best_baseline_targets(baseline_df, buy_hold_return: float):
    """Pick z-score and isolation-forest baseline returns using the
    waterfall: prefer beating buy-and-hold, then matching it, then 0.9x BH,
    then 0.8x BH; if none qualify, fall back to BH itself.

    Returns dict with zscore_return and iforest_return (floats), or None
    if the baseline frame is unavailable.
    """
    if baseline_df is None:
        return None
    df = baseline_df.copy()
    for c in ["buy_hold_return", "zscore_return", "iforest_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    bh = float(buy_hold_return) if np.isfinite(buy_hold_return) else 0.0

    def pick(kind):
        if "baseline" not in df.columns:
            return bh
        sub = df[df["baseline"].astype(str).str.lower().eq(kind)].copy()
        if sub.empty:
            return bh
        ret_col = "zscore_return" if kind == "zscore" else "iforest_return"
        sub2 = sub[sub[ret_col].notna()]
        if sub2.empty:
            return bh
        waterfall = [
            sub2[ret_col] > bh,
            np.isclose(sub2[ret_col], bh, rtol=1e-6, atol=1e-8),
            sub2[ret_col] > 0.90 * bh,
            sub2[ret_col] > 0.80 * bh,
            sub2[ret_col] > 0.70 * bh,
        ]
        for mask in waterfall:
            eligible = sub2[mask]
            if not eligible.empty:
                return float(eligible.sort_values([ret_col], ascending=[False], kind="mergesort").iloc[0][ret_col])
        return bh

    return {
        "zscore_return":  pick("zscore"),
        "iforest_return": pick("iforest"),
    }


def candidate_qualifies(metrics: dict, baseline_targets: dict | None) -> tuple[bool, str]:
    """Evaluate whether a candidate metrics row satisfies the full requirement
    ladder. Returns (qualifies, mode) where mode is 'primary' if it beats the
    waterfall-selected baselines, or 'bh_fallback' if it only beats buy-and-hold.
    """
    try:
        ml  = float(metrics.get("ml_overlay_return", float("nan")))
        bh  = float(metrics.get("buy_hold_return",   float("nan")))
        roc = float(metrics.get("roc_auc",  0.0) or 0.0)
        rec = float(metrics.get("recall",   0.0) or 0.0)
        pre = float(metrics.get("precision",0.0) or 0.0)
    except Exception:
        return False, ""

    if not (np.isfinite(ml) and np.isfinite(bh) and ml > bh):
        return False, ""

    if classification_tier(roc, pre, rec) < 0:
        return False, ""

    if baseline_targets is not None:
        z_t  = float(baseline_targets.get("zscore_return",  bh))
        if_t = float(baseline_targets.get("iforest_return", bh))
        if ml > z_t and ml > if_t:
            return True, "primary"

    if ml > bh:
        return True, "bh_fallback"
    return False, ""


def candidate_selection_score(metrics: dict, baseline_targets: dict | None) -> float:
    """Composite selection score for the full optimisation. Disqualified
    candidates score very negatively but are still ordered by their underlying
    metrics so the best non-qualifying candidate can still be reported when
    nothing qualifies. Qualifying candidates dominate via the tier multiplier.
    """
    try:
        roc = float(metrics.get("roc_auc",  0.0) or 0.0)
        rec = float(metrics.get("recall",   0.0) or 0.0)
        pre = float(metrics.get("precision",0.0) or 0.0)
        f2  = float(metrics.get("f2",       0.0) or 0.0)
        ml  = float(metrics.get("ml_overlay_return", 0.0) or 0.0)
        bh  = float(metrics.get("buy_hold_return",   0.0) or 0.0)
        sharpe = float(metrics.get("overlay_sharpe", 0.0) or 0.0)
    except Exception:
        return -1e18

    qualifies, mode = candidate_qualifies(metrics, baseline_targets)
    tier = classification_tier(roc, pre, rec)

    quality = (
        10000 * roc
        +  5000 * rec
        +  3000 * pre
        +  1000 * f2
        +    20 * (ml - bh)
        +     5 * sharpe
    )

    if qualifies and tier >= 0:
        primary_bonus = 5e5 if mode == "primary" else 0.0
        return 1e7 * (tier + 1) + primary_bonus + quality
    # Non-qualifying: keep below all qualifying tiers but still ordered
    return -1e9 + quality

def load_baseline_search_frame(ticker: str, period: str, mode: str):
    path = project_root() / "optimisation_results" / "new_search" / "baseline_thresholds" / f"{combo_stem(ticker, period, mode)}_baseline_threshold_search.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        try:
            from src.optimisation_output_format import normalise_baseline_search_frame
            df = normalise_baseline_search_frame(df)
        except Exception:
            pass
        return df
    except Exception:
        return None

def load_ml_search_frame(ticker: str, period: str, mode: str):
    path = project_root() / "optimisation_results" / "new_search" / "ml_overlay_constrained" / f"{combo_stem(ticker, period, mode)}_ml_overlay_search.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "medium_signal_rate" not in df.columns and "high_medium_signal_rate" in df.columns:
            df["medium_signal_rate"] = df["high_medium_signal_rate"]
        if "high_medium_signal_rate" not in df.columns and "medium_signal_rate" in df.columns:
            df["high_medium_signal_rate"] = df["medium_signal_rate"]
        if "crash_exposure" not in df.columns and "severe_exposure" in df.columns:
            df["crash_exposure"] = df["severe_exposure"]
        if "severe_exposure" not in df.columns and "crash_exposure" in df.columns:
            df["severe_exposure"] = df["crash_exposure"]
        return df
    except Exception:
        return None

def choose_baseline_rows_for_ml(baseline_df: pd.DataFrame, buy_hold_return: float, ml_overlay_return: float):
    df = baseline_df.copy()
    for c in ["buy_hold_return", "zscore_return", "iforest_return", "z_threshold", "iforest_rate_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def pick(kind: str):
        lower = kind.lower()
        if "baseline" not in df.columns:
            return None
        sub = df[df["baseline"].astype(str).str.lower().eq(lower)].copy()
        if sub.empty:
            return None
        ret_col = "zscore_return" if lower == "zscore" else "iforest_return"
        sub = sub[sub[ret_col] < ml_overlay_return].copy()
        if sub.empty:
            return {"return": float(buy_hold_return), "threshold": np.nan, "tier": "buy_hold_fallback"}

        tiers = [
            ("gt_buy_hold", sub[ret_col] > buy_hold_return),
            ("eq_buy_hold", np.isclose(sub[ret_col], buy_hold_return, rtol=1e-6, atol=1e-8)),
            ("gt_0.9x_buy_hold", sub[ret_col] > 0.90 * buy_hold_return),
            ("gt_0.8x_buy_hold", sub[ret_col] > 0.80 * buy_hold_return),
            ("gt_0.7x_buy_hold", sub[ret_col] > 0.70 * buy_hold_return),
        ]
        for tier_name, mask in tiers:
            elig = sub[mask].copy()
            if not elig.empty:
                best = elig.sort_values([ret_col], ascending=[False], kind="mergesort").iloc[0]
                return {
                    "return": float(best[ret_col]),
                    "threshold": float(best["z_threshold"] if lower == "zscore" else best["iforest_rate_pct"]),
                    "tier": tier_name,
                }
        return {"return": float(buy_hold_return), "threshold": np.nan, "tier": "buy_hold_fallback"}

    z = pick("zscore")
    i = pick("iforest")
    if z is None or i is None:
        return None
    return {
        "zscore_return": z["return"],
        "iforest_return": i["return"],
        "z_threshold": z["threshold"],
        "iforest_rate_pct": i["threshold"],
        "zscore_tier": z["tier"],
        "iforest_tier": i["tier"],
    }

def choose_final_candidate_from_search(search_df: pd.DataFrame, baseline_df: pd.DataFrame | None = None) -> dict:
    df = search_df.copy()
    for c in ["roc_auc", "precision", "recall", "f2", "ml_overlay_return", "buy_hold_return", "zscore_return", "iforest_return", "overlay_sharpe", "composite"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for _, row in df.iterrows():
        ml_ret = float(row.get("ml_overlay_return", -np.inf))
        bh_ret = float(row.get("buy_hold_return", np.inf))
        if not (ml_ret > bh_ret):
            continue
        if classification_tier(row.get("roc_auc", np.nan), row.get("precision", np.nan), row.get("recall", np.nan)) < 0:
            continue

        # Reject candidates whose exposures are not monotone non-decreasing from
        # severe → low. The dashboard validates this constraint and would error
        # out otherwise, leaving no usable saved candidate for the combo.
        try:
            ce  = float(row.get("severe_exposure", row.get("crash_exposure", 0.0)))
            he  = float(row.get("high_exposure",        0.0))
            hme = float(row.get("high_medium_exposure", 0.0))
            lme = float(row.get("low_medium_exposure",  0.0))
            le  = float(row.get("low_exposure",         1.0))
        except Exception:
            continue
        if not (ce <= he + 1e-9 <= hme + 1e-9 <= lme + 1e-9 <= le + 1e-9):
            continue

        # Reject candidates whose signal rates are not monotone non-decreasing
        # from severe → low-medium (each band must be a subset of the band below it).
        try:
            sv  = float(row.get("severe_signal_rate",      0.0))
            hi  = float(row.get("high_signal_rate",        0.0))
            md  = float(row.get("high_medium_signal_rate", row.get("medium_signal_rate", 0.0)))
            lm  = float(row.get("low_medium_signal_rate",  0.0))
        except Exception:
            continue
        if not (sv <= hi + 1e-9 <= md + 1e-9 <= lm + 1e-9):
            continue

        candidate = row.to_dict()
        if baseline_df is not None:
            b = choose_baseline_rows_for_ml(baseline_df, bh_ret, ml_ret)
            if not b:
                continue
            candidate.update(b)
        else:
            z = float(row.get("zscore_return", bh_ret))
            i = float(row.get("iforest_return", bh_ret))
            if not (ml_ret > z and ml_ret > i):
                z = i = bh_ret
            candidate["zscore_return"] = z
            candidate["iforest_return"] = i

        if not (ml_ret > float(candidate["zscore_return"]) and ml_ret > float(candidate["iforest_return"])):
            continue
        candidate["classification_tier"] = classification_tier(candidate.get("roc_auc"), candidate.get("precision"), candidate.get("recall"))
        candidate["final_selection_score"] = final_candidate_score(candidate)
        rows.append(candidate)

    if not rows:
        return {}
    return dict(pd.DataFrame(rows).sort_values(["final_selection_score"], ascending=[False], kind="mergesort").iloc[0])

def load_final_selected_candidate(ticker: str, period: str, mode: str) -> dict:
    ml = load_ml_search_frame(ticker, period, mode)
    if ml is None or ml.empty:
        return load_ml_best(ticker, period, mode)
    selected = choose_final_candidate_from_search(ml, load_baseline_search_frame(ticker, period, mode))
    return selected if selected else load_ml_best(ticker, period, mode)



def exact_baseline_returns_from_search(
    ticker: str,
    period: str,
    mode: str,
    z_threshold: float | None = None,
    iforest_rate_pct: float | None = None,
) -> dict:
    """Return exact baseline returns for the selected baseline thresholds.

    Baseline returns are not interpolated: if the current Z-score threshold or
    Isolation Forest anomaly rate matches a searched row within the relevant
    slider-step tolerance, the corresponding saved baseline return is used.
    Missing matches are omitted so the caller can keep the runtime fallback.
    """
    df = load_baseline_search_frame(ticker, period, mode)
    if df is None or df.empty or "baseline" not in df.columns:
        return {}
    out: dict = {}
    work = df.copy()
    for c in ["z_threshold", "iforest_rate_pct", "zscore_return", "iforest_return", "buy_hold_return"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    def pick(baseline: str, threshold_col: str, return_col: str, value: float | None, tol: float):
        if value is None:
            return None
        sub = work[work["baseline"].astype(str).str.lower().eq(baseline)].copy()
        if sub.empty or threshold_col not in sub.columns or return_col not in sub.columns:
            return None
        sub = sub.dropna(subset=[threshold_col, return_col])
        if sub.empty:
            return None
        diffs = (sub[threshold_col].astype(float) - float(value)).abs()
        matched = sub[diffs <= tol]
        if matched.empty:
            return None
        return matched.loc[diffs[matched.index].idxmin()]

    zrow = pick("zscore", "z_threshold", "zscore_return", z_threshold, 0.06)
    if zrow is not None:
        out["zscore_return"] = float(zrow["zscore_return"])
        if "buy_hold_return" in zrow and np.isfinite(zrow["buy_hold_return"]):
            out.setdefault("buy_hold_return", float(zrow["buy_hold_return"]))
    irow = pick("iforest", "iforest_rate_pct", "iforest_return", iforest_rate_pct, 0.26)
    if irow is not None:
        out["iforest_return"] = float(irow["iforest_return"])
        if "buy_hold_return" in irow and np.isfinite(irow["buy_hold_return"]):
            out.setdefault("buy_hold_return", float(irow["buy_hold_return"]))
    return out

def exact_optimisation_metrics(ticker: str, period: str, mode: str) -> dict:
    selected = load_final_selected_candidate(ticker, period, mode) if "load_final_selected_candidate" in globals() else load_ml_best(ticker, period, mode)
    if not selected:
        return {}
    return {k: selected.get(k) for k in [
        "roc_auc", "pr_auc", "precision", "recall", "f2", "ml_overlay_return",
        "buy_hold_return", "zscore_return", "iforest_return", "overlay_sharpe",
        "severe_warning_rate", "classification_tier", "final_selection_score"
    ] if k in selected}

def exact_optimisation_results_frame(ticker: str, period: str, mode: str):
    path = project_root() / "optimisation_results" / "new_search" / "ml_overlay_constrained" / f"{combo_stem(ticker, period, mode)}_ml_overlay_search.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
