"""
models.py  –  PulseGuard supervised models.

Ensemble: XGBoost + LightGBM (soft-voted probabilities, class-imbalance weighted).
Neural:   Lightweight TCN (Temporal Convolutional Network) trained only during
          full optimisation / manual retraining runs; loaded from disk cache on
          subsequent calls so that the main pipeline never re-trains it.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import random
from src.determinism import GLOBAL_RANDOM_SEED, configure_determinism, reset_random_seeds
configure_determinism()
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn
try:
    sklearn.set_config(assume_finite=False, working_memory=64)
except Exception:
    pass
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight



# ---------------------------------------------------------------------------
# Optional deep-learning imports – gracefully degrade if torch absent
# ---------------------------------------------------------------------------
try:
    if os.environ.get("PULSEGUARD_ENABLE_TORCH", "0") != "1":
        raise ImportError("PyTorch disabled for memory-safe dashboard startup")
    import torch
    # Streamlit may inspect torch.classes as a module path during startup.
    # Providing a harmless empty path prevents a non-fatal torch.classes warning
    # from being emitted in deployment logs.
    try:
        torch.classes.__path__ = []
    except Exception:
        pass
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    try:
        torch.manual_seed(GLOBAL_RANDOM_SEED)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# Optional gradient-boosting imports
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

# Path for persisting the trained TCN weights
TCN_CACHE_DIR = Path(__file__).resolve().parents[1] / ".pulseguard_tcn_cache"


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.70
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ---------------------------------------------------------------------------
# TCN neural classifier (lightweight, train-once-cache-forever)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _ResidualBlock(nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
            self.net = nn.Sequential(
                self.conv1, nn.ReLU(), nn.Dropout(0.1),
                self.conv2, nn.ReLU(), nn.Dropout(0.1),
            )
            self.skip = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            out = out[:, :, :x.size(2)]
            return self.relu(out + self.skip(x))

    class _TCNClassifier(nn.Module):
        def __init__(self, n_features, seq_len=20, hidden=32):
            super().__init__()
            self.seq_len = seq_len
            self.blocks = nn.Sequential(
                _ResidualBlock(n_features, hidden, kernel_size=3, dilation=1),
                _ResidualBlock(hidden, hidden, kernel_size=3, dilation=2),
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.blocks(x)
            h = h.mean(dim=2)
            return torch.sigmoid(self.head(h)).squeeze(1)


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    n, f = X.shape
    seqs = []
    for i in range(seq_len - 1, n):
        seqs.append(X[i - seq_len + 1: i + 1].T)
    return np.array(seqs, dtype=np.float32)


def _tcn_cache_key(n_features: int, seq_len: int, hidden: int, epochs: int) -> str:
    payload = json.dumps(
        {"n_features": n_features, "seq_len": seq_len, "hidden": hidden, "epochs": epochs},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def train_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seq_len: int = 20,
    hidden: int = 32,
    epochs: int = 30,
    lr: float = 3e-3,
    batch_size: int = 64,
):
    reset_random_seeds(GLOBAL_RANDOM_SEED)
    """Train a lightweight TCN and return the model (or None if torch unavailable)."""
    if not _TORCH_AVAILABLE:
        return None
    X_seq = _make_sequences(X_train, seq_len)
    y_seq = y_train[seq_len - 1:].astype(np.float32)
    pos_rate = float(y_seq.mean()) if len(y_seq) > 0 else 0.1
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], dtype=torch.float32)
    n, f = X_train.shape
    model = _TCNClassifier(n_features=f, seq_len=seq_len, hidden=hidden)
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.from_numpy(X_seq)
    y_t = torch.from_numpy(y_seq)
    model.train()
    for _ in range(epochs):
        generator = torch.Generator()
        generator.manual_seed(GLOBAL_RANDOM_SEED)
        perm = torch.randperm(len(X_t), generator=generator)
        for start in range(0, len(X_t), batch_size):
            idx = perm[start: start + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            pred = model(xb)
            weights = torch.where(yb == 1, pos_weight.expand_as(yb), torch.ones_like(yb))
            loss = (criterion(pred, yb) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_tcn(model, X: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Return probability array of length len(X); first seq_len-1 rows are NaN."""
    if not _TORCH_AVAILABLE or model is None:
        return np.full(len(X), np.nan)
    X_seq = _make_sequences(X, seq_len)
    with torch.no_grad():
        probs = model(torch.from_numpy(X_seq)).numpy()
    result = np.full(len(X), np.nan)
    result[seq_len - 1:] = probs
    return result


def save_tcn(model, cache_key: str) -> None:
    if model is None or not _TORCH_AVAILABLE:
        return
    TCN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = TCN_CACHE_DIR / f"tcn_{cache_key}.pkl"
    with path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _tcn_probabilities_for_frame(
    full_df: pd.DataFrame,
    train_index,
    feature_cols: list[str],
    seq_len: int = 20,
    hidden: int = 32,
    epochs: int = 20,
    train_if_missing: bool = False,
) -> np.ndarray:
    """Train/cache a TCN on train_index and return probabilities for full_df.

    The TCN is fitted only on the supplied training rows, then produces sequence
    probabilities for the whole frame. Rows before enough sequence history exists
    are returned as NaN and are ignored during blending.
    """
    probs = np.full(len(full_df), np.nan, dtype=float)

    if not _TORCH_AVAILABLE:
        return probs

    seq_len = int(max(5, min(seq_len, 120)))
    hidden = int(max(8, min(hidden, 128)))
    epochs = int(max(1, min(epochs, 100)))

    train = full_df.loc[train_index].dropna(subset=["future_risk_label"] + feature_cols).copy()
    if len(train) < max(60, seq_len + 20) or train["future_risk_label"].nunique() < 2:
        return probs

    try:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols].astype(float).fillna(0.0)).astype(np.float32)
        y_train = train["future_risk_label"].astype(int).to_numpy()

        cache_payload = json.dumps(
            {
                "columns": feature_cols,
                "train_start": str(train.index.min()),
                "train_end": str(train.index.max()),
                "n_train": int(len(train)),
                "seq_len": seq_len,
                "hidden": hidden,
                "epochs": epochs,
                "label_sum": int(y_train.sum()),
            },
            sort_keys=True,
        )
        cache_key = hashlib.sha256(cache_payload.encode()).hexdigest()[:16]
        model = load_tcn(cache_key)
        if model is None:
            if not train_if_missing:
                return probs
            model = train_tcn(X_train, y_train, seq_len=seq_len, hidden=hidden, epochs=epochs)
            save_tcn(model, cache_key)

        X_all = scaler.transform(full_df[feature_cols].astype(float).fillna(0.0)).astype(np.float32)
        probs = predict_tcn(model, X_all, seq_len=seq_len)
        return probs
    except Exception:
        return np.full(len(full_df), np.nan, dtype=float)


def _blend_with_tcn(
    ensemble_probs: np.ndarray,
    tcn_probs: np.ndarray,
    tcn_weight: float = 0.20,
) -> np.ndarray:
    """Blend ensemble and TCN probabilities while falling back to ensemble values."""
    ensemble_probs = np.asarray(ensemble_probs, dtype=float)
    tcn_probs = np.asarray(tcn_probs, dtype=float)
    weight = float(np.clip(tcn_weight, 0.0, 0.80))

    blended = ensemble_probs.copy()
    valid = ~np.isnan(tcn_probs)
    blended[valid] = (1.0 - weight) * ensemble_probs[valid] + weight * tcn_probs[valid]
    return blended


# ---------------------------------------------------------------------------
# XGBoost + LightGBM soft-voting ensemble
# ---------------------------------------------------------------------------

class XGBLGBEnsemble(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble of XGBoost and LightGBM with class-imbalance weighting."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        min_child_weight: int = 20,
        colsample_bytree: float = 0.70,
        learning_rate: float = 0.05,
        subsample: float = 0.80,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        random_state: int = GLOBAL_RANDOM_SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = int(random_state)
        self._xgb_model = None
        self._lgb_model = None
        self._fallback_model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        reset_random_seeds(self.random_state)
        y = np.asarray(y, dtype=int)
        sample_weights = compute_sample_weight("balanced", y)
        pos_rate = float(np.mean(y))
        scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)

        if _XGB_AVAILABLE:
            self._xgb_model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                colsample_bytree=self.colsample_bytree,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.random_state,
                seed=self.random_state,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
            )
            self._xgb_model.fit(X, y, sample_weight=sample_weights)

        if _LGB_AVAILABLE:
            self._lgb_model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_samples=self.min_child_weight,
                colsample_bytree=self.colsample_bytree,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                scale_pos_weight=scale_pos_weight,
                class_weight="balanced",
                random_state=self.random_state,
                seed=self.random_state,
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                bagging_seed=self.random_state,
                feature_fraction_seed=self.random_state,
                data_random_seed=self.random_state,
                drop_seed=self.random_state,
                verbose=-1,
            )
            self._lgb_model.fit(X, y, sample_weight=sample_weights)

        if not _XGB_AVAILABLE and not _LGB_AVAILABLE:
            from sklearn.ensemble import GradientBoostingClassifier
            self._fallback_model = GradientBoostingClassifier(
                n_estimators=min(self.n_estimators, 100),
                max_depth=self.max_depth,
                max_features=self.colsample_bytree,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            self._fallback_model.fit(X, y, sample_weight=sample_weights)

        return self

    def predict_proba(self, X):
        probs = []
        if self._xgb_model is not None:
            probs.append(self._xgb_model.predict_proba(X)[:, 1])
        if self._lgb_model is not None:
            probs.append(self._lgb_model.predict_proba(X)[:, 1])
        if self._fallback_model is not None:
            probs.append(self._fallback_model.predict_proba(X)[:, 1])
        avg = np.mean(probs, axis=0)
        return np.column_stack([1 - avg, avg])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Public training / prediction API
# ---------------------------------------------------------------------------

def fit_supervised_risk_model(
    train: pd.DataFrame,
    feature_cols: list[str],
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    max_features: float | str = 0.70,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    subsample: float = 0.80,
) -> Pipeline:
    reset_random_seeds(GLOBAL_RANDOM_SEED)
    """Train an XGBoost + LightGBM ensemble for crash risk prediction."""
    train = train.dropna(subset=["future_risk_label"]).copy()
    if len(train) < 50:
        raise ValueError("Not enough labelled training rows. Choose a longer timeframe.")
    if train["future_risk_label"].nunique() < 2:
        raise ValueError(
            "Training data only contains one outcome class. "
            "Use a longer timeframe or lower the risk threshold."
        )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBLGBEnsemble(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_samples_leaf,
            colsample_bytree=float(max_features) if isinstance(max_features, (float, int)) else 0.70,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
        )),
    ])
    model.fit(train[feature_cols], train["future_risk_label"].astype(int))
    return model


def f2_score(y_true, y_pred) -> float:
    return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    if len(y_true) == 0:
        return {
            "precision": float("nan"), "recall": float("nan"),
            "f1": float("nan"), "f2": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan"),
        }
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "f2":        float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")
    return out


def walk_forward_threshold_optimisation(
    probs: np.ndarray,
    labels: np.ndarray,
    min_precision: float = 0.60,
    min_roc_auc: float = 0.80,
) -> float:
    """Select threshold maximising F2-score with minimum precision & ROC-AUC constraints."""
    probs  = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask   = ~np.isnan(probs)
    probs, labels = probs[mask], labels[mask]
    if len(labels) == 0 or len(set(labels)) < 2:
        return float(np.quantile(probs, 0.95)) if len(probs) > 0 else 0.5
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    if auc < min_roc_auc:
        return float(np.quantile(probs, 0.85))
    best_thr, best_f2 = float(np.quantile(probs, 0.95)), -1.0
    for q in np.linspace(0.70, 0.98, 29):
        thr   = float(np.quantile(probs, q))
        preds = (probs >= thr).astype(int)
        if preds.sum() == 0:
            continue
        prec = float(precision_score(labels, preds, zero_division=0))
        if prec < min_precision:
            continue
        f2 = float(fbeta_score(labels, preds, beta=2, zero_division=0))
        if f2 > best_f2:
            best_f2 = f2
            best_thr = thr
    return best_thr


def predict_supervised_holdout(
    full_df: pd.DataFrame,
    train_fraction: float,
    feature_cols: list[str],
    probability_threshold: float,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    max_features: float | str = 0.70,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    subsample: float = 0.80,
    use_tcn: bool = True,
    tcn_seq_len: int = 20,
    tcn_hidden: int = 32,
    tcn_epochs: int = 20,
    tcn_weight: float = 0.20,
    train_tcn_if_missing: bool = False,
) -> tuple[pd.DataFrame, dict]:
    reset_random_seeds(GLOBAL_RANDOM_SEED)
    labelled = full_df.dropna(subset=["future_risk_label"]).copy()
    train_labelled, test_labelled = chronological_split(labelled, train_fraction=train_fraction)
    model = fit_supervised_risk_model(
        train_labelled, feature_cols,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_estimators=n_estimators,
        learning_rate=learning_rate, reg_lambda=reg_lambda,
        reg_alpha=reg_alpha, subsample=subsample,
    )
    out = full_df.copy()
    out["ml_risk_probability"] = np.nan
    out["ml_predicted_risk"] = False
    out["split"] = "unlabelled/latest"
    out.loc[train_labelled.index, "split"] = "train"
    out.loc[test_labelled.index, "split"] = "test"
    prediction_start_idx = train_labelled.index.max() + 1
    prediction_rows = out.loc[out.index >= prediction_start_idx].copy()
    ensemble_probs = model.predict_proba(prediction_rows[feature_cols])[:, 1]
    probs = ensemble_probs.copy()
    out["ensemble_risk_probability"] = np.nan
    out["tcn_risk_probability"] = np.nan

    if use_tcn:
        tcn_all = _tcn_probabilities_for_frame(
            out, train_labelled.index, feature_cols,
            seq_len=tcn_seq_len, hidden=tcn_hidden, epochs=tcn_epochs,
            train_if_missing=train_tcn_if_missing,
        )
        tcn_prediction = tcn_all[prediction_rows.index.to_numpy()]
        probs = _blend_with_tcn(ensemble_probs, tcn_prediction, tcn_weight=tcn_weight)
        out.loc[prediction_rows.index, "tcn_risk_probability"] = tcn_prediction

    preds = probs >= probability_threshold
    out.loc[prediction_rows.index, "ensemble_risk_probability"] = ensemble_probs
    out.loc[prediction_rows.index, "ml_risk_probability"] = probs
    out.loc[prediction_rows.index, "ml_predicted_risk"] = preds
    eval_rows = out.loc[test_labelled.index].dropna(
        subset=["future_risk_label", "ml_risk_probability"]
    )
    metrics = classification_metrics(
        eval_rows["future_risk_label"].astype(int),
        eval_rows["ml_predicted_risk"],
        eval_rows["ml_risk_probability"],
    )
    return out, metrics


def rolling_supervised_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    initial_train_size: int = 300,
    probability_threshold: float = 0.50,
    retrain_every: int = 20,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    max_features: float | str = 0.70,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    subsample: float = 0.80,
    use_tcn: bool = True,
    tcn_seq_len: int = 20,
    tcn_hidden: int = 32,
    tcn_epochs: int = 20,
    tcn_weight: float = 0.20,
    train_tcn_if_missing: bool = False,
) -> tuple[pd.DataFrame, dict]:
    reset_random_seeds(GLOBAL_RANDOM_SEED)
    out = df.copy()
    out["ml_risk_probability"] = np.nan
    out["ensemble_risk_probability"] = np.nan
    out["tcn_risk_probability"] = np.nan
    out["ml_predicted_risk"] = False
    out["split"] = "rolling_test"
    model = None
    for i in range(initial_train_size, len(out)):
        train = out.iloc[:i].dropna(subset=["future_risk_label"])
        if len(train) < 50 or train["future_risk_label"].nunique() < 2:
            continue
        if model is None or (i - initial_train_size) % retrain_every == 0:
            model = fit_supervised_risk_model(
                train, feature_cols,
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                max_features=max_features, n_estimators=n_estimators,
                learning_rate=learning_rate, reg_lambda=reg_lambda,
                reg_alpha=reg_alpha, subsample=subsample,
            )
        prob = model.predict_proba(out.iloc[[i]][feature_cols])[:, 1][0]
        out.loc[out.index[i], "ensemble_risk_probability"] = prob
        out.loc[out.index[i], "ml_risk_probability"] = prob
        out.loc[out.index[i], "ml_predicted_risk"] = prob >= probability_threshold
    if use_tcn:
        train_index = out.iloc[:initial_train_size].dropna(subset=["future_risk_label"]).index
        tcn_all = _tcn_probabilities_for_frame(
            out, train_index, feature_cols,
            seq_len=tcn_seq_len, hidden=tcn_hidden, epochs=tcn_epochs,
            train_if_missing=train_tcn_if_missing,
        )
        valid_blend = out["ensemble_risk_probability"].notna() & ~np.isnan(tcn_all)
        out.loc[valid_blend, "tcn_risk_probability"] = tcn_all[valid_blend.to_numpy()]
        blended = _blend_with_tcn(
            out.loc[valid_blend, "ensemble_risk_probability"].to_numpy(),
            out.loc[valid_blend, "tcn_risk_probability"].to_numpy(),
            tcn_weight=tcn_weight,
        )
        out.loc[valid_blend, "ml_risk_probability"] = blended
        out.loc[valid_blend, "ml_predicted_risk"] = blended >= probability_threshold

    valid = out["ml_risk_probability"].notna() & out["future_risk_label"].notna()
    metrics = classification_metrics(
        out.loc[valid, "future_risk_label"].astype(int),
        out.loc[valid, "ml_predicted_risk"],
        out.loc[valid, "ml_risk_probability"],
    )
    return out, metrics


def rolling_isolation_forest_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    initial_train_size: int = 300,
    contamination: float = 0.05,
    retrain_every: int = 70,
) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    out = df.copy()
    out["iforest_anomaly"] = False
    out["iforest_score"] = np.nan
    pipeline = None
    threshold = None
    initial_train_size = min(initial_train_size, max(60, len(out) // 2))
    for i in range(initial_train_size, len(out)):
        if pipeline is None or (i - initial_train_size) % retrain_every == 0:
            train = out.iloc[:i][feature_cols]
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", IsolationForest(
                    n_estimators=80, contamination=contamination,
                    random_state=GLOBAL_RANDOM_SEED, n_jobs=1,
                )),
            ])
            pipeline.fit(train)
            train_scaled = pipeline.named_steps["scaler"].transform(train)
            train_scores = -pipeline.named_steps["model"].score_samples(train_scaled)
            threshold = np.quantile(train_scores, 1 - contamination)
        x = out.iloc[[i]][feature_cols]
        x_scaled = pipeline.named_steps["scaler"].transform(x)
        score = -pipeline.named_steps["model"].score_samples(x_scaled)[0]
        out.loc[out.index[i], "iforest_score"] = score
        out.loc[out.index[i], "iforest_anomaly"] = bool(score >= threshold)
    return out


def zscore_baseline(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    out["zscore_anomaly"] = out["zscore"].abs() >= threshold
    out["zscore_score"] = out["zscore"].abs()
    return out
