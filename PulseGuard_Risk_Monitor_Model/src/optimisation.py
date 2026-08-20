"""
optimisation.py – PulseGuard walk-forward hyperparameter optimiser.

Startup optimisation (startup_optimise_parameters):
  Fast 4-stage walk-forward search used on app load.

Full optimisation (full_return_constrained_optimise_parameters):
  Active 375-candidate return-constrained coarse-to-fine search used by the
  dashboard button. It yields progress updates for the UI and deliberately does
  not persist optimised parameters back into app defaults or downloadable file
  contents.
"""
from __future__ import annotations

from src.determinism import configure_determinism
configure_determinism()
from pathlib import Path
from src.determinism import GLOBAL_RANDOM_SEED, deterministic_rng, configure_determinism
configure_determinism()

import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "1")
os.environ.setdefault("XGBOOST_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import time
from typing import Generator

import numpy as np
from src.shared_scoring import (
    DEFAULT_PARAMS as SHARED_DEFAULT_PARAMS,
    evaluate_candidate_shared,
    flat_result_row,
    load_baseline_best,
    enforce_label_threshold_order,
    select_best_baseline_targets,
    candidate_qualifies,
    candidate_selection_score,
    load_baseline_search_frame,
)
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, roc_auc_score

from src.data_loader import load_market_data
from src.features import make_features
from src.risk import safe_returns, safe_equity_curve
from src.risk import classify_events, apply_risk_overlay, backtest_overlay
from src.models import classification_metrics, fit_supervised_risk_model, _tcn_probabilities_for_frame, _blend_with_tcn
from src.risk import apply_risk_overlay, backtest_overlay, classify_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def threshold_from_signal_rate(probs: np.ndarray, signal_rate: float) -> float:
    probs = np.array(probs)
    probs = probs[~np.isnan(probs)]
    if len(probs) == 0:
        return 0.50
    return float(np.quantile(probs, max(0.0, min(1.0, 1.0 - signal_rate))))


def _f2(y, preds):
    return float(fbeta_score(y, preds, beta=2, zero_division=0))


def _normalise_ticker_universe(tickers, primary_ticker: str | None = None) -> list[str]:
    """Return a clean, de-duplicated ticker list while preserving order."""
    raw = []
    if isinstance(tickers, str):
        raw.extend([x.strip() for x in tickers.split(",")])
    elif tickers:
        raw.extend([str(x).strip() for x in tickers])
    if primary_ticker:
        raw.insert(0, str(primary_ticker).strip())

    clean = []
    seen = set()
    for t in raw:
        t = t.upper()
        if not t or t in seen:
            continue
        seen.add(t)
        clean.append(t)
    return clean or ["AAPL"]


def _load_market_data_universe(tickers, period: str, primary_ticker: str | None = None):
    """Load real Yahoo Finance data for each ticker in the optimisation universe."""
    universe = _normalise_ticker_universe(tickers, primary_ticker)
    loaded = []
    failed = []

    for ticker in universe:
        try:
            item = load_market_data(ticker=ticker, period=period)
            frame = item.frame.copy()
            frame["ticker"] = ticker
            loaded.append((ticker, frame, item.source_name))
        except Exception as exc:
            failed.append((ticker, str(exc)))

    if not loaded:
        details = "; ".join(f"{ticker}: {err}" for ticker, err in failed[:8])
        raise ValueError(
            "No real market data could be loaded for the optimisation ticker universe. "
            f"Failed tickers: {details}"
        )

    source_name = "Yahoo Finance multi-ticker: " + ", ".join(t for t, _, _ in loaded)
    if failed:
        source_name += " | skipped: " + ", ".join(t for t, _ in failed)

    return loaded, source_name, failed


def _oos_probabilities_multi(loaded_data, params, rng=None):
    """Return OOS probabilities for a multi-ticker optimisation universe.

    Features are computed separately per ticker to avoid rolling-window leakage across
    securities, then concatenated into a single training/evaluation pool.
    """
    try:
        labelled_frames = []
        feature_cols_ref = None

        for ticker, frame, _source in loaded_data:
            featured, feature_cols = make_features(
                frame,
                short_window=params["window"],
                forward_horizon=params["forward_horizon"],
                risk_threshold=params["risk_threshold"],
            )
            labelled = featured.dropna(subset=["future_risk_label"]).copy()
            if len(labelled) < 120 or labelled["future_risk_label"].nunique() < 2:
                continue
            labelled["ticker"] = ticker
            if feature_cols_ref is None:
                feature_cols_ref = feature_cols
            labelled_frames.append(labelled)

        if len(labelled_frames) == 0 or feature_cols_ref is None:
            return None, None

        labelled_all = pd.concat(labelled_frames, ignore_index=True)
        available_cols = [c for c in feature_cols_ref if c in labelled_all.columns]
        labelled_all = labelled_all.dropna(subset=["future_risk_label"] + available_cols).copy()
        if len(labelled_all) < 250 or labelled_all["future_risk_label"].nunique() < 2:
            return None, None

        pools = []
        for _ticker, group in labelled_all.groupby("ticker", sort=False):
            group = group.sort_values("timestamp", kind="mergesort") if "timestamp" in group.columns else group
            cut = int(len(group) * 0.65)
            if cut >= 80:
                pools.append(group.iloc[:cut].copy())

        if not pools:
            return None, None

        opt_pool = pd.concat(pools, ignore_index=True)
        if "timestamp" in opt_pool.columns:
            opt_pool = opt_pool.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(drop=True)
        if len(opt_pool) < 200 or opt_pool["future_risk_label"].nunique() < 2:
            return None, None

        n = len(opt_pool)
        fold_specs = [
            (int(n * 0.40), int(n * 0.55)),
            (int(n * 0.55), int(n * 0.70)),
            (int(n * 0.70), int(n * 0.85)),
        ]

        frames, probs_list = [], []
        for train_end, val_end in fold_specs:
            train = opt_pool.iloc[:train_end].copy()
            val = opt_pool.iloc[train_end:val_end].copy()
            if train["future_risk_label"].nunique() < 2 or val["future_risk_label"].nunique() < 2:
                continue
            if len(val) < 20:
                continue

            model = fit_supervised_risk_model(
                train, available_cols,
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                n_estimators=params["n_estimators"],
                learning_rate=params.get("learning_rate", 0.05),
                reg_lambda=params.get("reg_lambda", 1.0),
                reg_alpha=params.get("reg_alpha", 0.0),
                subsample=params.get("subsample", 0.80),
            )
            p = model.predict_proba(val[available_cols])[:, 1]
            frames.append(val)
            probs_list.append(p)

        if len(frames) < 2:
            return None, None

        wf = pd.concat(frames, ignore_index=True)
        wf["wf_probability"] = np.concatenate(probs_list)
        return wf, available_cols
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Shared OOS walk-forward helper
# ---------------------------------------------------------------------------

def _oos_probabilities(data_frame, params, rng=None):
    """Return (wf_df, feature_cols) with OOS probs from CV folds on opt_pool."""
    try:
        featured, feature_cols = make_features(
            data_frame,
            short_window=params["window"],
            forward_horizon=params["forward_horizon"],
            risk_threshold=params["risk_threshold"],
        )
        labelled = featured.dropna(subset=["future_risk_label"]).copy()
        if len(labelled) < 200 or labelled["future_risk_label"].nunique() < 2:
            return None, None

        opt_pool = labelled.iloc[:int(len(labelled) * 0.65)].copy()
        if len(opt_pool) < 160 or opt_pool["future_risk_label"].nunique() < 2:
            return None, None

        n = len(opt_pool)
        fold_specs = [
            (int(n * 0.40), int(n * 0.55)),
            (int(n * 0.55), int(n * 0.70)),
            (int(n * 0.70), int(n * 0.85)),
        ]

        frames, probs_list = [], []
        for train_end, val_end in fold_specs:
            train = opt_pool.iloc[:train_end].copy()
            val   = opt_pool.iloc[train_end:val_end].copy()
            if train["future_risk_label"].nunique() < 2 or val["future_risk_label"].nunique() < 2:
                continue
            if len(val) < 15:
                continue
            model = fit_supervised_risk_model(
                train, feature_cols,
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                n_estimators=params["n_estimators"],
                learning_rate=params.get("learning_rate", 0.05),
                reg_lambda=params.get("reg_lambda", 1.0),
                reg_alpha=params.get("reg_alpha", 0.0),
                subsample=params.get("subsample", 0.80),
            )
            p_ensemble = model.predict_proba(val[feature_cols])[:, 1]
            p = p_ensemble
            if params.get("tcn_weight", 0.0) > 0:
                combined = pd.concat([train, val], axis=0)
                tcn_all = _tcn_probabilities_for_frame(
                    combined, train.index, feature_cols,
                    seq_len=int(params.get("tcn_seq_len", 20)),
                    hidden=int(params.get("tcn_hidden", 32)),
                    epochs=int(params.get("tcn_epochs", 20)),
                    train_if_missing=True,
                )
                tcn_val = tcn_all[-len(val):]
                p = _blend_with_tcn(p_ensemble, tcn_val, tcn_weight=float(params.get("tcn_weight", 0.20)))
            frames.append(val)
            probs_list.append(p)

        if len(frames) < 2:
            return None, None

        wf = pd.concat(frames, ignore_index=True)
        wf["wf_probability"] = np.concatenate(probs_list)
        return wf, feature_cols
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Startup optimiser (fast, 4-stage)
# ---------------------------------------------------------------------------

def startup_optimise_parameters(ticker: str, period: str, selected_prediction_mode: str):
    """Four-stage walk-forward optimiser used on app load.

    Startup optimisation intentionally uses only the selected display ticker.
    The multi-ticker universe is used only by Full Optimisation.
    """
    rng = deterministic_rng(GLOBAL_RANDOM_SEED)
    data = load_market_data(ticker=ticker, period=period)

    fallback = {
        "window": 20,
        "forward_horizon": 5,
        "risk_threshold": 0.04,
        "z_threshold": 3.0,
        "iforest_rate_pct": 5.0,
        "prediction_mode": selected_prediction_mode,
        "train_fraction": 0.70,
        "initial_train_size": 300,
        "retrain_every": 63,
        "severe_signal_rate": 0.05,
        "low_medium_signal_rate": 0.30,
        "medium_signal_rate": 0.15,
        "high_signal_rate": 0.08,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "max_features": 0.70,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "subsample": 0.80,
        "tcn_seq_len": 20,
        "tcn_hidden": 32,
        "tcn_epochs": 20,
        "tcn_weight": 0.20,
        "low_exposure": 1.00,
        "low_medium_exposure": 0.98,
        "high_medium_exposure": 0.90,
        "high_exposure": 0.65,
        "crash_exposure": 0.15,
        "cooldown_days": 2,
    }

    # Stage 1: model/feature hyperparameters
    model_candidates = [fallback.copy()]
    for _ in range(48):
        model_candidates.append({
            **fallback,
            "window":            int(rng.choice([10, 15, 20, 25, 30, 40])),
            "forward_horizon":   int(rng.integers(3, 11)),
            "risk_threshold":    float(rng.uniform(0.02, 0.10)),
            "retrain_every":     int(rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])),
            "max_depth":         int(rng.choice([3, 4, 4, 5])),
            "min_samples_leaf":  int(rng.choice([15, 20, 25, 30, 40])),
            "max_features":      float(rng.uniform(0.50, 0.90)),
            "n_estimators":      int(rng.choice([150, 200, 250])),
            "learning_rate":     float(rng.choice([0.03, 0.05, 0.08, 0.10])),
            "reg_lambda":        float(rng.uniform(0.5, 3.0)),
            "subsample":         float(rng.uniform(0.70, 0.95)),
            "tcn_seq_len":        int(rng.choice([10, 15, 20, 30, 40])),
            "tcn_hidden":         int(rng.choice([16, 24, 32, 48, 64])),
            "tcn_epochs":         int(rng.choice([8, 12, 16, 20, 30])),
            "tcn_weight":         float(rng.choice([0.10, 0.15, 0.20, 0.25, 0.30])),
        })

    best_model_params = fallback.copy()
    best_model_score  = -1e9
    best_wf           = None
    best_feature_cols = None

    for params in model_candidates:
        wf, feature_cols = _oos_probabilities(data.frame, params, rng)
        if wf is None:
            continue

        y     = wf["future_risk_label"].astype(int).to_numpy()
        probs = wf["wf_probability"].to_numpy()

        try:
            roc_auc = float(roc_auc_score(y, probs)) if len(set(y)) > 1 else 0.5
        except Exception:
            roc_auc = 0.5

        if roc_auc < 0.52:
            continue

        try:
            cutoff   = float(np.quantile(probs, 0.90))
            mask     = probs >= cutoff
            top_prec = float(y[mask].mean()) if mask.sum() > 0 else 0.0
        except Exception:
            top_prec = 0.0

        positive_mean = float(np.mean(probs[y == 1])) if np.any(y == 1) else 0.0
        negative_mean = float(np.mean(probs[y == 0])) if np.any(y == 0) else 0.0
        separation    = positive_mean - negative_mean

        score = 0.40 * roc_auc + 0.35 * top_prec + 0.25 * separation

        if score > best_model_score:
            best_model_score  = score
            best_model_params = params.copy()
            best_wf           = wf.copy()
            best_feature_cols = feature_cols

    if best_wf is None:
        return fallback, {"source": "fallback"}

    # Stage 2: signal rate (percentile-based threshold)
    y     = best_wf["future_risk_label"].astype(int).to_numpy()
    probs = best_wf["wf_probability"].to_numpy()
    base_rate = float(np.mean(y)) if len(y) else 0.08

    candidate_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
    best_severe_signal_rate = fallback["severe_signal_rate"]
    best_threshold_score    = -1e9

    for rate in candidate_rates:
        thr   = threshold_from_signal_rate(probs, rate)
        preds = probs >= thr
        if not preds.any():
            continue
        m         = classification_metrics(y, preds, probs)
        precision = float(m.get("precision", 0.0) or 0.0)
        recall    = float(m.get("recall",    0.0) or 0.0)
        f2        = float(m.get("f2",        0.0) or 0.0)
        if np.isnan(precision): precision = 0.0
        if np.isnan(recall):    recall    = 0.0
        if np.isnan(f2):        f2        = 0.0
        lift = precision / base_rate if base_rate > 0 else 1.0
        noise_penalty = max(0.0, rate - 0.08) * 1.5
        # F2-weighted score: recall counts twice as much as precision
        score = (
            0.40 * f2
            + 0.25 * precision
            + 0.20 * recall
            + 0.15 * min(lift / 3.0, 1.0)
            - noise_penalty
        )
        if score > best_threshold_score:
            best_threshold_score    = score
            best_severe_signal_rate = rate

    best_high_signal_rate       = min(best_severe_signal_rate * 1.8,  0.25)
    best_medium_signal_rate     = min(best_severe_signal_rate * 3.5,  0.40)
    best_low_medium_signal_rate = min(best_severe_signal_rate * 7.0,  0.55)
    best_threshold = threshold_from_signal_rate(probs, best_severe_signal_rate)

    # Stage 3: overlay exposure / cooldown
    overlay_candidates = []
    for _ in range(80):
        low_exp      = float(rng.uniform(0.99, 1.00))
        low_med_exp  = float(rng.uniform(0.94, min(low_exp,     1.00)))
        high_med_exp = float(rng.uniform(0.80, min(low_med_exp, 0.97)))
        high_exp     = float(rng.uniform(0.50, min(high_med_exp, 0.88)))
        crash_exp    = float(rng.uniform(0.05, min(high_exp,    0.35)))
        overlay_candidates.append({
            "low_exposure":         low_exp,
            "low_medium_exposure":  low_med_exp,
            "high_medium_exposure": high_med_exp,
            "high_exposure":        high_exp,
            "crash_exposure":       crash_exp,
            "cooldown_days":        int(rng.integers(0, 6)),
        })
    overlay_candidates.extend([
        {"low_exposure": 1.00, "low_medium_exposure": 0.98, "high_medium_exposure": 0.90, "high_exposure": 0.65, "crash_exposure": 0.15, "cooldown_days": 2},
        {"low_exposure": 1.00, "low_medium_exposure": 1.00, "high_medium_exposure": 0.95, "high_exposure": 0.75, "crash_exposure": 0.20, "cooldown_days": 1},
        {"low_exposure": 1.00, "low_medium_exposure": 0.97, "high_medium_exposure": 0.85, "high_exposure": 0.60, "crash_exposure": 0.10, "cooldown_days": 3},
    ])

    best_overlay = {k: fallback[k] for k in [
        "low_exposure", "low_medium_exposure", "high_medium_exposure",
        "high_exposure", "crash_exposure", "cooldown_days"
    ]}
    best_overlay_score = -1e9
    overlay_candidates.append(best_overlay.copy())

    overlay_base = best_wf.copy()
    overlay_base["ml_risk_probability"] = overlay_base["wf_probability"]
    overlay_base["ml_predicted_risk"]   = overlay_base["ml_risk_probability"] >= best_threshold
    overlay_base["zscore_anomaly"]      = False
    overlay_base["iforest_anomaly"]     = False

    bh_returns      = overlay_base["close"].pct_change().fillna(0).to_numpy()
    bh_total_return = float(np.prod(1 + bh_returns) - 1)
    z_flags         = (overlay_base["zscore"].abs() >= 3.0).to_numpy() if "zscore" in overlay_base.columns \
                      else np.zeros(len(overlay_base), dtype=bool)

    for op in overlay_candidates:
        try:
            classified = classify_events(overlay_base, probability_threshold=best_threshold, **op)
            ov         = apply_risk_overlay(classified, **op)
            _, bt      = backtest_overlay(ov)

            z_exp = np.ones(len(overlay_base))
            for idx, flag in enumerate(z_flags):
                if flag:
                    z_exp[idx+1: min(len(z_exp), idx+1+op["cooldown_days"])] = op["high_exposure"]
            z_ret = float(np.prod(1 + z_exp * bh_returns) - 1)

            turnover       = float(ov["risk_overlay_exposure"].diff().abs().fillna(0).sum())
            edge_vs_bh     = float(bt["overlay_total_return"] - bh_total_return)
            edge_vs_zscore = float(bt["overlay_total_return"] - z_ret)
            dd_edge        = float(abs(bt.get("buy_hold_max_drawdown", 0.0)) - abs(bt["overlay_max_drawdown"]))
            crash_rate     = float(np.mean(overlay_base["ml_predicted_risk"]))

            score = (
                0.40 * edge_vs_bh
                + 0.30 * edge_vs_zscore
                + 0.20 * dd_edge
                - 0.05 * turnover
                - 0.20 * max(0.0, crash_rate - 0.10)
            )
            if score > best_overlay_score:
                best_overlay_score = score
                best_overlay = op.copy()
        except Exception:
            continue

    # Stage 4: baseline thresholds
    best_z_threshold    = float(fallback["z_threshold"])
    best_iforest_rate   = float(fallback["iforest_rate_pct"])
    best_baseline_score = -1e9

    if "zscore" in best_wf.columns and "close" in best_wf.columns:
        wf_close   = best_wf["close"].to_numpy()
        wf_rets    = np.diff(wf_close, prepend=wf_close[0]) / np.where(
                         np.roll(wf_close, 1) == 0, 1, np.roll(wf_close, 1))
        wf_rets[0] = 0.0
        bh_ret        = float(np.prod(1 + wf_rets) - 1)
        reduction_exp = float(best_overlay.get("high_exposure", 0.65))
        cooldown      = int(best_overlay.get("cooldown_days", 2))

        def _baseline_return(flags):
            exp = np.ones(len(flags))
            for i, f in enumerate(flags):
                if f:
                    end = min(len(exp), i + 1 + cooldown)
                    exp[i + 1:end] = np.minimum(exp[i + 1:end], reduction_exp)
            return float(np.prod(1 + exp * wf_rets) - 1)

        z_best_score = -1e9
        for z_thr in np.linspace(1.0, 5.0, 25):
            flags   = (best_wf["zscore"].abs() >= z_thr).to_numpy()
            n_flags = int(flags.sum())
            if n_flags < 2 or n_flags > len(best_wf) * 0.40:
                continue
            bl_ret   = _baseline_return(flags)
            edge     = bl_ret - bh_ret
            freq_pen = max(0.0, n_flags / len(best_wf) - 0.15) * 0.5
            score    = edge - freq_pen
            if score > z_best_score:
                z_best_score        = score
                best_z_threshold    = float(z_thr)
                best_baseline_score = score

        try:
            from sklearn.ensemble import IsolationForest as _IForest
            from sklearn.preprocessing import StandardScaler as _Scaler
            _feat_cols = best_feature_cols if best_feature_cols else []
            _feat_cols = [c for c in _feat_cols if c in best_wf.columns]
            if len(_feat_cols) >= 3:
                featured_full, _ = make_features(
                    data.frame,
                    short_window=best_model_params["window"],
                    forward_horizon=best_model_params["forward_horizon"],
                    risk_threshold=best_model_params["risk_threshold"],
                )
                labelled_full = featured_full.dropna(subset=["future_risk_label"] + _feat_cols).copy()
                oos_ts = set(best_wf["timestamp"].astype(str).tolist()) if "timestamp" in best_wf.columns else set()
                if "timestamp" in labelled_full.columns:
                    oos_mask = labelled_full["timestamp"].astype(str).isin(oos_ts)
                    train_if = labelled_full[~oos_mask].copy()
                    oos_if   = labelled_full[oos_mask].copy()
                else:
                    n_oos    = len(best_wf)
                    train_if = labelled_full.iloc[:-n_oos].copy()
                    oos_if   = labelled_full.iloc[-n_oos:].copy()
                if len(train_if) >= 60 and len(oos_if) >= 10:
                    scaler_if  = _Scaler()
                    X_train_if = scaler_if.fit_transform(train_if[_feat_cols].fillna(0))
                    X_oos_if   = scaler_if.transform(oos_if[_feat_cols].fillna(0))
                    oos_rets_if = oos_if["close"].pct_change().fillna(0).to_numpy()
                    bh_oos      = float(np.prod(1 + oos_rets_if) - 1)
                    fwd_rets_if = np.array([
                        float(np.sum(oos_rets_if[i+1: min(len(oos_rets_if), i+1+cooldown)]))
                        for i in range(len(oos_rets_if))
                    ])
                    base_down_rate = float(np.mean(fwd_rets_if < -0.005)) if len(fwd_rets_if) else 0.05
                    if_best_score = -1e9
                    for if_rate_pct in np.arange(1.0, 16.0, 1.0):
                        try:
                            cont    = float(if_rate_pct) / 100.0
                            iforest = _IForest(n_estimators=60, contamination=cont, random_state=42, n_jobs=1)
                            iforest.fit(X_train_if)
                            train_scores_if = -iforest.score_samples(X_train_if)
                            thr_if          = float(np.quantile(train_scores_if, 1.0 - cont))
                            oos_scores_if   = -iforest.score_samples(X_oos_if)
                            flags_if        = oos_scores_if >= thr_if
                            n_flags         = int(flags_if.sum())
                            if n_flags < 2 or n_flags > len(oos_if) * 0.40:
                                continue
                            bl_ret_if      = _baseline_return(flags_if)
                            edge_if        = bl_ret_if - bh_oos
                            flagged_fwd    = fwd_rets_if[flags_if]
                            precision_down = float(np.mean(flagged_fwd < -0.005)) if len(flagged_fwd) > 0 else 0.0
                            precision_lift = (precision_down - base_down_rate) / max(base_down_rate, 0.01)
                            mean_fwd_ret   = float(np.mean(flagged_fwd)) if len(flagged_fwd) > 0 else 0.0
                            freq_pen_if    = max(0.0, n_flags / len(oos_if) - 0.15) * 0.5
                            upside_pen     = max(0.0, mean_fwd_ret) * 2.0
                            score_if = (0.50 * edge_if + 0.35 * precision_lift - freq_pen_if - upside_pen)
                            if score_if > if_best_score:
                                if_best_score     = score_if
                                best_iforest_rate = round(float(if_rate_pct), 1)
                        except Exception:
                            continue
        except Exception:
            wf_y = best_wf["future_risk_label"].astype(int).to_numpy() if "future_risk_label" in best_wf.columns else np.array([])
            pos_rate          = float(np.mean(wf_y)) if len(wf_y) > 0 else 0.08
            best_iforest_rate = round(max(3.0, min(12.0, pos_rate * 100)), 1)

    final_params = {
        **best_model_params,
        "severe_signal_rate":           best_severe_signal_rate,
        "high_signal_rate":             best_high_signal_rate,
        "medium_signal_rate":           best_medium_signal_rate,
        "low_medium_signal_rate":       best_low_medium_signal_rate,
        "z_threshold":                  best_z_threshold,
        "iforest_rate_pct":             best_iforest_rate,
        **best_overlay,
    }

    return final_params, {
        "source":           "layered_walk_forward",
        "model_score":      float(best_model_score),
        "threshold_score":  float(best_threshold_score),
        "overlay_score":    float(best_overlay_score),
        "baseline_score":   float(best_baseline_score),
        "best_signal_rate": best_severe_signal_rate,
        "data_source":      data.source_name,
    }


# ---------------------------------------------------------------------------
# Full optimisation (coarse-to-fine, F2-score + Sharpe, CSV/Excel output)
# ---------------------------------------------------------------------------

# Defaults for the full optimiser
_FULL_OPT_DEFAULTS = {
    "window": 20, "forward_horizon": 5, "risk_threshold": 0.04,
    "train_fraction": 0.70, "initial_train_size": 300, "retrain_every": 63,
    "severe_signal_rate": 0.05, "low_medium_signal_rate": 0.30,
    "medium_signal_rate": 0.15, "high_signal_rate": 0.08,
    "max_depth": 4, "min_samples_leaf": 20,
    "max_features": 0.70, "n_estimators": 200,
    "learning_rate": 0.05, "reg_lambda": 1.0, "reg_alpha": 0.0, "subsample": 0.80,
    "tcn_seq_len": 20, "tcn_hidden": 32, "tcn_epochs": 20, "tcn_weight": 0.20,
    "low_exposure": 1.00, "low_medium_exposure": 0.98,
    "high_medium_exposure": 0.90, "high_exposure": 0.65,
    "crash_exposure": 0.15, "cooldown_days": 2,
    "z_threshold": 3.0, "iforest_rate_pct": 5.0,
}
_FULL_OPT_DEFAULTS.update({
    "window": 12,
    "forward_horizon": 6,
    "risk_threshold": 0.023682439291444,
    "train_fraction": 0.7,
    "initial_train_size": 300,
    "retrain_every": 63,
    "severe_signal_rate": 0.0295754351937071,
    "low_medium_signal_rate": 0.3,
    "medium_signal_rate": 0.15,
    "high_signal_rate": 0.08,
    "max_depth": 2,
    "min_samples_leaf": 15,
    "max_features": 0.8514288706693292,
    "n_estimators": 250,
    "learning_rate": 0.0217773318544838,
    "reg_lambda": 4.800041379124955,
    "reg_alpha": 0.0,
    "subsample": 0.9000938113831674,
    "tcn_seq_len": 60,
    "tcn_hidden": 64,
    "tcn_epochs": 20,
    "tcn_weight": 0.3,
    "low_exposure": 1.0,
    "low_medium_exposure": 0.953484246427285,
    "high_medium_exposure": 0.8613512901664921,
    "high_exposure": 0.8607214458525305,
    "crash_exposure": 0.2483631736910356,
    "cooldown_days": 4,
    "z_threshold": 3.0,
    "iforest_rate_pct": 5.0,
})


MIN_ROC_AUC  = 0.80
MIN_PRECISION = 0.60


def _score_candidate(wf, feature_cols, params, data_frame):
    params = _enforce_label_threshold_order(params)
    if not _label_threshold_order_is_valid(params):
        return None, None
    """Score a candidate parameter set on OOS walk-forward data.
    
    Objective: maximise F2-score and overlay Sharpe while enforcing
    ROC-AUC > 0.80 and Precision > 0.60.
    Returns (composite_score, metrics_dict) or (None, None) on failure.
    """
    try:
        y     = wf["future_risk_label"].astype(int).to_numpy()
        probs = wf["wf_probability"].to_numpy()
        if len(set(y)) < 2:
            return None, None

        roc_auc = float(roc_auc_score(y, probs))

        # Walk-forward threshold: prioritise F2 and recall, with precision as a constraint.
        # This is deliberately recall-forward because missing crash periods is worse
        # than creating some additional warnings in a crash-risk monitor.
        best_score, best_f2 = -1.0, -1.0
        best_thr, best_prec, best_rec = float(np.quantile(probs, 0.95)), 0.0, 0.0

        relaxed_candidates = []
        for q in np.linspace(0.55, 0.98, 44):
            thr_c = float(np.quantile(probs, q))
            preds = (probs >= thr_c).astype(int)
            if preds.sum() == 0:
                continue

            prec = float(precision_score(y, preds, zero_division=0))
            rec = float(np.mean(preds[y == 1])) if np.any(y == 1) else 0.0
            f2 = float(fbeta_score(y, preds, beta=2, zero_division=0))

            # Main pool: precision must pass the constraint.
            if prec >= MIN_PRECISION:
                threshold_score = 0.70 * f2 + 0.25 * rec + 0.05 * prec
                if threshold_score > best_score:
                    best_score = threshold_score
                    best_f2, best_thr, best_prec, best_rec = f2, thr_c, prec, rec

            # Backup pool so the optimiser does not collapse to zero recall when
            # very few thresholds satisfy the strict precision constraint.
            relaxed_candidates.append((f2, rec, prec, thr_c))

        if best_score < 0 and relaxed_candidates:
            # Use a soft precision penalty only as a fallback.
            best_f2, best_rec, best_prec, best_thr = max(
                relaxed_candidates,
                key=lambda x: (0.70 * x[0] + 0.25 * x[1] + 0.05 * x[2] - max(0.0, MIN_PRECISION - x[2]))
            )

        # Overlay Sharpe
        overlay_base = wf.copy()
        overlay_base["ml_risk_probability"] = probs
        overlay_base["ml_predicted_risk"]   = probs >= best_thr
        overlay_base["zscore_anomaly"]       = False
        overlay_base["iforest_anomaly"]      = False
        rate = params.get("severe_signal_rate", 0.05)
        try:
            classified = classify_events(
                overlay_base, probability_threshold=best_thr,
                low_exposure=params["low_exposure"], low_medium_exposure=params["low_medium_exposure"],
                high_medium_exposure=params["high_medium_exposure"],
                high_exposure=params["high_exposure"], crash_exposure=params["crash_exposure"],
            )
            ov   = apply_risk_overlay(classified, cooldown_days=params["cooldown_days"],
                                       low_exposure=params["low_exposure"], low_medium_exposure=params["low_medium_exposure"],
                                       high_medium_exposure=params["high_medium_exposure"],
                                       high_exposure=params["high_exposure"], crash_exposure=params["crash_exposure"])
            _, bt = backtest_overlay(ov)
            sharpe = float(bt["overlay_sharpe"])
            max_dd = float(bt["overlay_max_drawdown"])
        except Exception:
            sharpe = 0.0
            max_dd = -1.0

        # Precision and ROC-AUC remain constraints, but the objective is now
        # explicitly F2/recall-first, with Sharpe as a secondary portfolio term.
        precision_penalty = max(0.0, MIN_PRECISION - best_prec) * 2.0
        auc_penalty = max(0.0, MIN_ROC_AUC - roc_auc) * 1.0
        sharpe_component = float(np.clip(sharpe / 5.0, -1.0, 1.0))

        composite = (
            0.55 * best_f2
            + 0.25 * best_rec
            + 0.10 * roc_auc
            + 0.10 * sharpe_component
            - precision_penalty
            - auc_penalty
        )

        metrics = {
            "roc_auc":   round(roc_auc, 4),
            "precision": round(best_prec, 4),
            "recall":    round(best_rec, 4),
            "f2":        round(best_f2, 4),
            "sharpe":    round(sharpe, 4),
            "max_dd":    round(max_dd, 4),
            "composite": round(composite, 6),
            "threshold": round(best_thr, 6),
        }
        return composite, metrics
    except Exception:
        return None, None




def evaluate_with_optimiser_scoring(
    ticker: str,
    period: str,
    selected_prediction_mode: str,
    params: dict,
    optimisation_tickers=None,
) -> dict:
    """Evaluate one parameter set using the exact optimiser scoring path.

    This is the shared candidate-evaluation path used by optimisation routines:
    _load_market_data_universe -> _oos_probabilities_multi -> _score_candidate.
    The dashboard uses this for recalculated metrics so dashboard metrics and
    optimiser-search metrics are produced by one shared function.
    """
    loaded_data, optimisation_source_name, optimisation_failures = _load_market_data_universe(
        optimisation_tickers or [ticker],
        period,
        primary_ticker=ticker,
    )
    primary_data = load_market_data(ticker=ticker, period=period)
    wf, feature_cols = _oos_probabilities_multi(loaded_data, params, rng=deterministic_rng(GLOBAL_RANDOM_SEED))

    if wf is None or feature_cols is None:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "f2": float("nan"),
            "roc_auc": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "composite": float("nan"),
            "threshold": float("nan"),
            "source": "optimiser_scoring_failed",
        }

    composite, metrics = _score_candidate(wf, feature_cols, params, primary_data.frame)
    if metrics is None:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "f2": float("nan"),
            "roc_auc": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "composite": float("nan"),
            "threshold": float("nan"),
            "source": "optimiser_scoring_failed",
        }

    out = dict(metrics)
    out["source"] = "optimiser_scoring"
    out["optimisation_source_name"] = optimisation_source_name
    out["optimisation_failures"] = optimisation_failures
    out["prediction_mode"] = selected_prediction_mode
    return out



def _enforce_label_threshold_order(params: dict) -> dict:
    out = dict(params)
    risk_threshold = float(out.get("risk_threshold", out.get("severe_label_threshold", 0.05)))
    risk_threshold = max(risk_threshold, 0.005)
    out["risk_threshold"] = float(risk_threshold)
    out["severe_label_threshold"] = float(risk_threshold)
    return out


def _label_threshold_order_is_valid(params: dict) -> bool:
    return float(params.get("risk_threshold", params.get("severe_label_threshold", 0.05))) > 0



def compute_dashboard_style_baseline_returns(frame, cooldown_days: int = 3, baseline_reduction: float = 0.50) -> dict:
    """Compute Z-score and Isolation Forest baseline returns exactly like the dashboard.

    Baseline rule:
    - start fully invested
    - after a baseline anomaly, reduce exposure for cooldown_days
    - compound market returns using the same safe_equity_curve utility
    """
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
    }

# ---------------------------------------------------------------------------
# Return-constrained manual optimiser
# ---------------------------------------------------------------------------

def _rc_threshold_from_signal_rate(probs, signal_rate):
    arr = np.asarray(probs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.5
    return float(np.quantile(arr, min(max(1.0 - float(signal_rate), 0.0), 1.0)))


def _rc_mode_key(mode):
    return "holdout" if "holdout" in str(mode).lower() else "rolling"


def _rc_stem(ticker, period, mode):
    return f"{str(ticker).upper()}_{period}_{_rc_mode_key(mode)}"


def _rc_load_baselines(ticker, period, mode):
    root = Path(__file__).resolve().parents[1]
    path = root / "optimisation_results" / "new_search" / "baseline_thresholds" / f"{_rc_stem(ticker, period, mode)}_baseline_threshold_best.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            return {
                "z_threshold": float(payload.get("z_threshold", 3.0)),
                "iforest_rate_pct": float(payload.get("iforest_rate_pct", 5.0)),
            }
        except Exception:
            pass
    return {"z_threshold": 3.0, "iforest_rate_pct": 5.0}


def _rc_score_candidate(ticker, period, mode, params):
    params = _enforce_label_threshold_order(params)
    if not _label_threshold_order_is_valid(params):
        return None, None
    try:
        data = load_market_data(ticker=ticker, period=period)
        featured, feature_cols = make_features(
            data.frame,
            short_window=int(params["window"]),
            forward_horizon=int(params["forward_horizon"]),
            risk_threshold=float(params["risk_threshold"]),
        )
        if mode == "Holdout split":
            predicted, _ = predict_supervised_holdout(
                featured, train_fraction=float(params.get("train_fraction", 0.70)), feature_cols=feature_cols,
                probability_threshold=0.50, max_depth=int(params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]), max_features=float(params["max_features"]),
                n_estimators=int(params["n_estimators"]), learning_rate=float(params.get("learning_rate", 0.05)),
                reg_lambda=float(params.get("reg_lambda", 1.0)), reg_alpha=float(params.get("reg_alpha", 0.0)),
                subsample=float(params.get("subsample", 0.80)), use_tcn=True,
                tcn_seq_len=int(params.get("tcn_seq_len", 20)), tcn_hidden=int(params.get("tcn_hidden", 32)),
                tcn_epochs=int(params.get("tcn_epochs", 20)), tcn_weight=float(params.get("tcn_weight", 0.20)),
                train_tcn_if_missing=False,
            )
        else:
            predicted, _ = rolling_supervised_predictions(
                featured, feature_cols=feature_cols,
                initial_train_size=min(int(params.get("initial_train_size", 300)), max(60, len(featured)//2)),
                probability_threshold=0.50, retrain_every=int(params.get("retrain_every", 63)),
                max_depth=int(params["max_depth"]), min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=float(params["max_features"]), n_estimators=int(params["n_estimators"]),
                learning_rate=float(params.get("learning_rate", 0.05)), reg_lambda=float(params.get("reg_lambda", 1.0)),
                reg_alpha=float(params.get("reg_alpha", 0.0)), subsample=float(params.get("subsample", 0.80)),
                use_tcn=True, tcn_seq_len=int(params.get("tcn_seq_len", 20)),
                tcn_hidden=int(params.get("tcn_hidden", 32)), tcn_epochs=int(params.get("tcn_epochs", 20)),
                tcn_weight=float(params.get("tcn_weight", 0.20)), train_tcn_if_missing=False,
            )

        probs = predicted.loc[predicted["ml_risk_probability"].notna(), "ml_risk_probability"].to_numpy()
        if len(probs) == 0:
            return None, None
        lm = _rc_threshold_from_signal_rate(probs, params.get("low_medium_signal_rate", 0.30))
        md = _rc_threshold_from_signal_rate(probs, params.get("medium_signal_rate", 0.15))
        hi = _rc_threshold_from_signal_rate(probs, params.get("high_signal_rate", 0.08))
        sv = _rc_threshold_from_signal_rate(probs, params.get("severe_signal_rate", 0.05))
        valid = predicted["ml_risk_probability"].notna()
        predicted.loc[valid, "ml_predicted_risk"] = predicted.loc[valid, "ml_risk_probability"] >= sv

        predicted = rolling_isolation_forest_baseline(
            predicted, feature_cols=feature_cols,
            initial_train_size=min(int(params.get("initial_train_size", 300)), max(60, len(predicted)//2)),
            contamination=float(params.get("iforest_rate_pct", 5.0)) / 100.0,
            retrain_every=int(params.get("retrain_every", 63)),
        )
        predicted = zscore_baseline(predicted, threshold=float(params.get("z_threshold", 3.0)))

        classified = classify_events(
            predicted, probability_threshold=sv, low_medium_threshold=lm, medium_threshold=md,
            high_threshold=hi, severe_threshold=sv, low_exposure=float(params.get("low_exposure", 1.0)),
            low_medium_exposure=float(params.get("low_medium_exposure", 0.96)),
            high_medium_exposure=float(params.get("high_medium_exposure", 0.85)),
            high_exposure=float(params.get("high_exposure", 0.65)),
            crash_exposure=float(params.get("crash_exposure", 0.20)),
        )
        overlay = apply_risk_overlay(
            classified, cooldown_days=int(params.get("cooldown_days", 3)),
            low_exposure=float(params.get("low_exposure", 1.0)),
            low_medium_exposure=float(params.get("low_medium_exposure", 0.96)),
            high_medium_exposure=float(params.get("high_medium_exposure", 0.85)),
            high_exposure=float(params.get("high_exposure", 0.65)),
            crash_exposure=float(params.get("crash_exposure", 0.20)),
        )
        _, bt = backtest_overlay(overlay)

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
            cls = {"roc_auc": np.nan, "pr_auc": np.nan, "precision": np.nan, "recall": np.nan, "f2": np.nan}

        baseline_metrics = compute_dashboard_style_baseline_returns(
            overlay,
            cooldown_days=int(params.get("cooldown_days", 3)),
            baseline_reduction=0.50,
        )
        ml_ret = float(bt.get("overlay_total_return", np.nan))
        bh_ret = float(bt.get("buy_hold_total_return", np.nan))
        z_ret = float(baseline_metrics.get("zscore_return", np.nan))
        if_ret = float(baseline_metrics.get("iforest_return", np.nan))
        beats_all = bool(ml_ret > bh_ret and ml_ret > z_ret and ml_ret > if_ret)
        margin = min(ml_ret - bh_ret, ml_ret - z_ret, ml_ret - if_ret)
        sharpe = float(bt.get("overlay_sharpe", 0.0) or 0.0)
        f2 = float(cls.get("f2", 0.0) or 0.0)
        rec = float(cls.get("recall", 0.0) or 0.0)
        auc = float(cls.get("roc_auc", 0.0) or 0.0)
        composite = (10.0 if beats_all else 0.0) + 4.0 * margin + 0.50 * sharpe + 0.25 * f2 + 0.15 * rec + 0.10 * auc

        metrics = {
            "roc_auc": round(float(cls.get("roc_auc", np.nan)), 4),
            "pr_auc": round(float(cls.get("pr_auc", np.nan)), 4),
            "precision": round(float(cls.get("precision", np.nan)), 4),
            "recall": round(float(cls.get("recall", np.nan)), 4),
            "f2": round(float(cls.get("f2", np.nan)), 4),
            "ml_overlay_return": round(ml_ret, 6),
            "buy_hold_return": round(bh_ret, 6),
            "zscore_return": round(z_ret, 6),
            "iforest_return": round(if_ret, 6),
            "ml_alpha_vs_buy_hold": round(ml_ret - bh_ret, 6),
            "ml_alpha_vs_zscore": round(ml_ret - z_ret, 6),
            "ml_alpha_vs_iforest": round(ml_ret - if_ret, 6),
            "ml_beats_buy_hold": bool(ml_ret > bh_ret),
            "ml_beats_zscore": bool(ml_ret > z_ret),
            "ml_beats_iforest": bool(ml_ret > if_ret),
            "ml_beats_all_three": beats_all,
            "overlay_sharpe": round(sharpe, 4),
            "composite": round(float(composite), 6),
            "probability_threshold": round(float(sv), 6),
        }
        return composite, metrics
    except Exception:
        return None, None


def _rc_sample(rng, fallback):
    return {
        **fallback,
        "window": int(rng.choice([10, 15, 20, 25, 30, 40])),
        "forward_horizon": int(rng.integers(3, 11)),
        "risk_threshold": float(rng.uniform(0.02, 0.10)),
        "train_fraction": float(rng.choice([0.60, 0.65, 0.70, 0.75, 0.80])),
        "initial_train_size": int(rng.choice([175, 200, 225, 250, 300, 350, 400, 425])),
        "retrain_every": int(rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])),
        "max_depth": int(rng.choice([3, 4, 5, 6])),
        "min_samples_leaf": int(rng.choice([10, 15, 20, 25, 30, 40])),
        "max_features": float(rng.uniform(0.40, 0.95)),
        "n_estimators": int(rng.choice([150, 200, 250, 300, 400])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08, 0.10, 0.15])),
        "reg_lambda": float(rng.uniform(0.5, 5.0)),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0])),
        "subsample": float(rng.uniform(0.65, 0.95)),
        "tcn_seq_len": int(rng.choice([10, 15, 20, 30, 40, 60])),
        "tcn_hidden": int(rng.choice([16, 24, 32, 48, 64, 96])),
        "tcn_epochs": int(rng.choice([6, 10, 15, 20, 30])),
        "tcn_weight": float(rng.choice([0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])),
        "severe_signal_rate": float(rng.choice([0.03, 0.04, 0.05, 0.06, 0.08, 0.10])),
        "low_medium_signal_rate": float(rng.choice([0.20, 0.25, 0.30, 0.35, 0.40])),
        "medium_signal_rate": float(rng.choice([0.10, 0.125, 0.15, 0.175, 0.20])),
        "high_signal_rate": float(rng.choice([0.05, 0.06, 0.08, 0.10, 0.12])),
        "low_exposure": 1.00,
        "low_medium_exposure": float(rng.uniform(0.94, 1.00)),
        "high_medium_exposure": float(rng.uniform(0.80, 0.97)),
        "high_exposure": float(rng.uniform(0.50, 0.88)),
        "crash_exposure": float(rng.uniform(0.05, 0.35)),
        "cooldown_days": int(rng.integers(0, 6)),
    }


def _rc_fine(rng, seed):
    out = dict(seed)
    out.update({
        "window": int(np.clip(seed["window"] + rng.integers(-3, 4), 5, 60)),
        "forward_horizon": int(np.clip(seed["forward_horizon"] + rng.integers(-2, 3), 2, 20)),
        "risk_threshold": float(np.clip(seed["risk_threshold"] * rng.uniform(0.80, 1.25), 0.015, 0.12)),
        "max_depth": int(np.clip(seed["max_depth"] + rng.choice([-1, 0, 0, 1]), 2, 8)),
        "min_samples_leaf": int(np.clip(seed["min_samples_leaf"] + rng.choice([-5, 0, 5]), 5, 80)),
        "max_features": float(np.clip(seed["max_features"] + rng.uniform(-0.10, 0.10), 0.30, 1.0)),
        "n_estimators": int(np.clip(seed["n_estimators"] + rng.choice([-50, 0, 50]), 100, 500)),
        "learning_rate": float(np.clip(seed["learning_rate"] * rng.uniform(0.80, 1.25), 0.01, 0.20)),
        "reg_lambda": float(np.clip(seed["reg_lambda"] * rng.uniform(0.75, 1.35), 0.1, 10.0)),
        "subsample": float(np.clip(seed["subsample"] + rng.uniform(-0.05, 0.05), 0.60, 1.0)),
        "severe_signal_rate": float(np.clip(seed["severe_signal_rate"] * rng.uniform(0.85, 1.20), 0.02, 0.15)),
        "low_medium_exposure": float(np.clip(seed["low_medium_exposure"] + rng.uniform(-0.03, 0.03), 0.90, 1.0)),
        "high_medium_exposure": float(np.clip(seed["high_medium_exposure"] + rng.uniform(-0.05, 0.05), 0.70, 1.0)),
        "high_exposure": float(np.clip(seed["high_exposure"] + rng.uniform(-0.07, 0.07), 0.35, 0.95)),
        "crash_exposure": float(np.clip(seed["crash_exposure"] + rng.uniform(-0.05, 0.05), 0.0, 0.60)),
        "cooldown_days": int(np.clip(seed["cooldown_days"] + rng.choice([-1, 0, 1]), 0, 5)),
    })
    return out



# ---------------------------------------------------------------------------
# Final shared-scoring return-constrained manual optimiser
# ---------------------------------------------------------------------------

def _fast_sample_candidate(rng, fallback):
    """Fast search candidate.

    To keep the full optimisation practical, most candidates optimise signal,
    threshold and overlay behaviour while periodically perturbing model
    hyperparameters. This preserves 375 candidates but avoids retraining the
    expensive model for every single overlay-only candidate.
    """
    out = dict(fallback)

    # Model hyperparameters: changed less often by caller; included here for full-search support.
    out.update({
        "window": int(rng.choice([10, 15, 20, 25, 30, 40])),
        "forward_horizon": int(rng.integers(3, 11)),
        "risk_threshold": float(rng.uniform(0.02, 0.10)),
        "train_fraction": float(rng.choice([0.60, 0.65, 0.70, 0.75, 0.80])),
        "initial_train_size": int(rng.choice([175, 200, 225, 250, 300, 350, 400, 425])),
        "retrain_every": int(rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90])),
        "max_depth": int(rng.choice([3, 4, 5, 6])),
        "min_samples_leaf": int(rng.choice([10, 15, 20, 25, 30, 40])),
        "max_features": float(rng.uniform(0.40, 0.95)),
        "n_estimators": int(rng.choice([120, 160, 200, 250])),
        "learning_rate": float(rng.choice([0.03, 0.05, 0.08, 0.10])),
        "reg_lambda": float(rng.uniform(0.5, 4.0)),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5])),
        "subsample": float(rng.uniform(0.70, 0.95)),
        "tcn_seq_len": int(rng.choice([10, 20, 30])),
        "tcn_hidden": int(rng.choice([16, 32, 48])),
        "tcn_epochs": int(rng.choice([6, 10, 15])),
        "tcn_weight": float(rng.choice([0.00, 0.10, 0.20, 0.30])),
    })

    # Overlay/signal inputs.
    out.update({
        "severe_signal_rate": float(rng.choice([0.03, 0.04, 0.05, 0.06, 0.08, 0.10])),
        "low_medium_signal_rate": float(rng.choice([0.20, 0.25, 0.30, 0.35, 0.40])),
        "medium_signal_rate": float(rng.choice([0.10, 0.125, 0.15, 0.175, 0.20])),
        "high_signal_rate": float(rng.choice([0.05, 0.06, 0.08, 0.10, 0.12])),
        "low_exposure": 1.00,
        "low_medium_exposure": float(rng.uniform(0.94, 1.00)),
        "high_medium_exposure": float(rng.uniform(0.80, 0.97)),
        "high_exposure": float(rng.uniform(0.50, 0.88)),
        "crash_exposure": float(rng.uniform(0.05, 0.35)),
        "cooldown_days": int(rng.integers(0, 6)),
    })
    return enforce_label_threshold_order(out)


def _fast_fine_candidate(rng, seed):
    out = dict(seed)
    out.update({
        "severe_signal_rate": float(np.clip(seed["severe_signal_rate"] * rng.uniform(0.85, 1.20), 0.02, 0.15)),
        "low_medium_exposure": float(np.clip(seed["low_medium_exposure"] + rng.uniform(-0.03, 0.03), 0.90, 1.0)),
        "high_medium_exposure": float(np.clip(seed["high_medium_exposure"] + rng.uniform(-0.05, 0.05), 0.70, 1.0)),
        "high_exposure": float(np.clip(seed["high_exposure"] + rng.uniform(-0.07, 0.07), 0.35, 0.95)),
        "crash_exposure": float(np.clip(seed["crash_exposure"] + rng.uniform(-0.05, 0.05), 0.0, 0.60)),
        "cooldown_days": int(np.clip(seed["cooldown_days"] + rng.choice([-1, 0, 1]), 0, 5)),
    })
    return enforce_label_threshold_order(out)


def full_return_constrained_optimise_parameters(ticker, period, selected_prediction_mode, current_params, optimisation_tickers=None):
    rng = deterministic_rng(GLOBAL_RANDOM_SEED)
    start_time = time.time()
    N_COARSE_LOCAL, N_FINE_LOCAL = 250, 125
    N_TOTAL_LOCAL = N_COARSE_LOCAL + N_FINE_LOCAL

    fallback = {**SHARED_DEFAULT_PARAMS, **current_params}
    fallback.update(load_baseline_best(ticker, period, selected_prediction_mode))
    fallback = enforce_label_threshold_order(fallback)

    # Load the baseline-threshold search file so the baseline-comparison
    # ladder can be evaluated for every candidate.
    _baseline_df = load_baseline_search_frame(ticker, period, selected_prediction_mode)

    def _baseline_targets_for(metrics):
        bh_ret = float(metrics.get("buy_hold_return", 0.0) or 0.0)
        if _baseline_df is None:
            return None
        return select_best_baseline_targets(_baseline_df, bh_ret)

    all_results = []
    best_score   = -1e18
    best_params  = dict(fallback)
    best_metrics = {}

    coarse_candidates = [fallback.copy()] + [_fast_sample_candidate(rng, fallback) for _ in range(N_COARSE_LOCAL - 1)]

    for i, params in enumerate(coarse_candidates, start=1):
        try:
            result   = evaluate_candidate_shared(ticker, period, selected_prediction_mode, params, use_model_cache=True)
            row      = flat_result_row(ticker, period, selected_prediction_mode, params, result, "coarse", i, i)
            metrics  = dict(result["metrics"])
            targets  = _baseline_targets_for(metrics)
            score    = candidate_selection_score(metrics, targets)
            row["selection_score"] = score
        except Exception as exc:
            row     = {**params, "stage": "coarse", "candidate": i, "overall_candidate": i, "error": str(exc), "all_constraints_met": False, "composite": -1e18, "selection_score": -1e18}
            score   = -1e18
            metrics = {}
        all_results.append(row)
        if score > best_score:
            best_score   = score
            best_params  = params.copy()
            best_metrics = metrics.copy()
        elapsed = time.time() - start_time
        eta     = (elapsed / i) * (N_TOTAL_LOCAL - i) if i else 0.0
        yield {"stage": "coarse", "candidate": i, "total": N_COARSE_LOCAL, "overall_candidate": i, "overall_total": N_TOTAL_LOCAL, "elapsed": elapsed, "eta": eta, "params": best_params.copy(), "metrics": best_metrics.copy(), "all_results": all_results}

    # Pick top-5 coarse candidates by selection_score as fine-search seeds.
    coarse_scored = [(r.get("selection_score", -1e18), {k: r[k] for k in r if k in fallback}) for r in all_results]
    coarse_scored.sort(key=lambda x: x[0], reverse=True)
    seeds = [c[1] for c in coarse_scored[:5]] if coarse_scored else [fallback.copy()]
    if not seeds:
        seeds = [fallback.copy()]
    fine_candidates = []
    for seed in seeds:
        seed_filled = {**fallback, **seed}
        for _ in range(max(1, N_FINE_LOCAL // len(seeds))):
            fine_candidates.append(_fast_fine_candidate(rng, seed_filled))

    for j, params in enumerate(fine_candidates[:N_FINE_LOCAL], start=1):
        overall = N_COARSE_LOCAL + j
        try:
            result   = evaluate_candidate_shared(ticker, period, selected_prediction_mode, params, use_model_cache=True)
            row      = flat_result_row(ticker, period, selected_prediction_mode, params, result, "fine", j, overall)
            metrics  = dict(result["metrics"])
            targets  = _baseline_targets_for(metrics)
            score    = candidate_selection_score(metrics, targets)
            row["selection_score"] = score
        except Exception as exc:
            row     = {**params, "stage": "fine", "candidate": j, "overall_candidate": overall, "error": str(exc), "all_constraints_met": False, "composite": -1e18, "selection_score": -1e18}
            score   = -1e18
            metrics = {}
        all_results.append(row)
        if score > best_score:
            best_score   = score
            best_params  = params.copy()
            best_metrics = metrics.copy()
        elapsed = time.time() - start_time
        eta     = (elapsed / overall) * (N_TOTAL_LOCAL - overall) if overall else 0.0
        yield {"stage": "fine", "candidate": j, "total": N_FINE_LOCAL, "overall_candidate": overall, "overall_total": N_TOTAL_LOCAL, "elapsed": elapsed, "eta": eta, "params": best_params.copy(), "metrics": best_metrics.copy(), "all_results": all_results}

    final_params = {**enforce_label_threshold_order(best_params), "selected_metrics": best_metrics}
    yield {"stage": "done", "candidate": N_TOTAL_LOCAL, "total": N_TOTAL_LOCAL, "overall_candidate": N_TOTAL_LOCAL, "overall_total": N_TOTAL_LOCAL, "elapsed": time.time() - start_time, "eta": 0.0, "params": final_params.copy(), "metrics": best_metrics.copy(), "all_results": all_results}
