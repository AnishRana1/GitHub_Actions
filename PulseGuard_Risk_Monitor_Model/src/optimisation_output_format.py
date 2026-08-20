from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BLANK_A = ["", " ", "  "]
BLANK_B = ["   ", "    ", "     "]
BLANK_C = ["      ", "       ", "        "]
BLANK_D = ["         ", "          ", "           "]
BLANK_E = ["            ", "             ", "              "]


def _num(x: Any, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    try:
        return bool(x)
    except Exception:
        return False


def _blank_dict(cols: list[str]) -> dict[str, Any]:
    return {c: np.nan for c in cols}


def _metric_flags(roc: Any, recall: Any, precision: Any) -> dict[str, bool]:
    roc = _num(roc, -np.inf)
    rec = _num(recall, -np.inf)
    pre = _num(precision, -np.inf)
    return {
        "classification_requirement_met_1": bool(roc >= 0.60 and rec >= 0.20 and pre >= 0.20),
        "classification_requirement_met_2": bool(roc >= 0.60 and rec >= 0.20 and pre >= 0.15),
        "classification_requirement_met_3": bool(roc >= 0.60 and rec >= 0.15 and pre >= 0.20),
        "classification_requirement_met_4": bool(roc >= 0.60 and rec >= 0.15 and pre >= 0.15),
        "classification_requirement_met_5": bool(roc >= 0.57 and rec >= 0.15 and pre >= 0.15),
        "classification_requirement_met_6": bool(roc >= 0.54 and rec >= 0.15 and pre >= 0.15),
    }


def _get_col(df: pd.DataFrame, names: list[str], default=np.nan):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _baseline_return_series(df: pd.DataFrame, baseline: pd.Series) -> pd.Series:
    if "zscore_or_iforest_return" in df.columns:
        return pd.to_numeric(df["zscore_or_iforest_return"], errors="coerce")
    if "zscore_return_or_iforest_return" in df.columns:
        return pd.to_numeric(df["zscore_return_or_iforest_return"], errors="coerce")
    if "baseline_return" in df.columns:
        return pd.to_numeric(df["baseline_return"], errors="coerce")
    z = pd.to_numeric(_get_col(df, ["zscore_return"], np.nan), errors="coerce")
    i = pd.to_numeric(_get_col(df, ["iforest_return"], np.nan), errors="coerce")
    return pd.Series(np.where(baseline.eq("zscore"), z, i), index=df.index)


def _baseline_threshold_series(df: pd.DataFrame, baseline: pd.Series) -> pd.Series:
    if "z_threshold_or_iforest_rate_pct" in df.columns:
        return pd.to_numeric(df["z_threshold_or_iforest_rate_pct"], errors="coerce")
    if "baseline_threshold" in df.columns:
        return pd.to_numeric(df["baseline_threshold"], errors="coerce")
    z = pd.to_numeric(_get_col(df, ["z_threshold"], np.nan), errors="coerce")
    i = pd.to_numeric(_get_col(df, ["iforest_rate_pct"], np.nan), errors="coerce")
    return pd.Series(np.where(baseline.eq("zscore"), z, i), index=df.index)


def format_baseline_search_results(df: pd.DataFrame) -> pd.DataFrame:
    """Presentation format for baseline-threshold search files."""
    if df is None or df.empty:
        return pd.DataFrame()
    raw = df.copy()
    if "stage_candidate" in raw.columns and "candidate" not in raw.columns:
        raw["candidate"] = raw["stage_candidate"]
    if "baseline" in raw.columns:
        base = raw["baseline"].astype(str).str.lower().str.replace("z-score", "zscore", regex=False)
    else:
        base = pd.Series(["zscore" if i < 50 else "iforest" for i in range(len(raw))], index=raw.index)
    base = base.where(base.isin(["zscore", "iforest"]), pd.Series(["zscore" if i < 50 else "iforest" for i in range(len(raw))], index=raw.index))
    raw["_baseline_norm"] = base
    raw["_baseline_threshold"] = _baseline_threshold_series(raw, base)
    raw["_baseline_return"] = _baseline_return_series(raw, base)

    ordered = []
    for b in ["zscore", "iforest"]:
        part = raw[raw["_baseline_norm"].eq(b)].copy()
        if not part.empty:
            ordered.append(part)
    raw = pd.concat(ordered, ignore_index=True) if ordered else raw.reset_index(drop=True)

    rows = []
    for idx, r in raw.iterrows():
        baseline = str(r.get("_baseline_norm", "zscore" if idx < 50 else "iforest")).lower()
        stage_candidate = int(_num(r.get("candidate", r.get("stage_candidate", (idx % 50) + 1)), (idx % 50) + 1))
        overall_candidate = int(_num(r.get("overall_candidate", idx + 1), idx + 1))
        buy_hold = _num(r.get("buy_hold_return"))
        ret = _num(r.get("_baseline_return"))
        beats_bh = bool(np.isfinite(ret) and np.isfinite(buy_hold) and ret > buy_hold)
        equals_bh = bool(np.isfinite(ret) and np.isfinite(buy_hold) and np.isclose(ret, buy_hold, rtol=1e-6, atol=1e-8))
        beats_09 = bool(np.isfinite(ret) and np.isfinite(buy_hold) and ret > 0.90 * buy_hold)
        beats_08 = bool(np.isfinite(ret) and np.isfinite(buy_hold) and ret > 0.80 * buy_hold)
        beats_07 = bool(np.isfinite(ret) and np.isfinite(buy_hold) and ret > 0.70 * buy_hold)
        row = {
            "overall_candidate": overall_candidate,
            "stage": r.get("stage", baseline),
            "stage_candidate": stage_candidate,
            **_blank_dict(BLANK_A),
            "baseline": baseline,
            "z_threshold_or_iforest_rate_pct": _num(r.get("_baseline_threshold")),
            **_blank_dict(BLANK_B),
            "buy_hold_return": buy_hold,
            "zscore_or_iforest_return": ret,
            **_blank_dict(BLANK_C),
            "zscore_or_iforest_beats_buy_hold": beats_bh,
            "zscore_or_iforest_equals_buy_hold": equals_bh,
            "zscore_or_iforest_beats_0.9x_buy_hold": beats_09,
            "zscore_or_iforest_beats_0.8x_buy_hold": beats_08,
            "zscore_or_iforest_beats_0.7x_buy_hold": beats_07,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def normalise_baseline_search_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a code-friendly baseline frame from either old or presentation CSVs."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "stage_candidate" in out.columns and "candidate" not in out.columns:
        out["candidate"] = out["stage_candidate"]
    if "baseline" not in out.columns:
        out["baseline"] = ["zscore" if i < 50 else "iforest" for i in range(len(out))]
    base = out["baseline"].astype(str).str.lower().str.replace("z-score", "zscore", regex=False)
    out["baseline"] = base
    if "z_threshold" not in out.columns or "iforest_rate_pct" not in out.columns:
        vals = _baseline_threshold_series(out, base)
        out["z_threshold"] = np.where(base.eq("zscore"), vals, np.nan)
        out["iforest_rate_pct"] = np.where(base.eq("iforest"), vals, np.nan)
    if "zscore_return" not in out.columns or "iforest_return" not in out.columns:
        vals = _baseline_return_series(out, base)
        out["zscore_return"] = np.where(base.eq("zscore"), vals, np.nan)
        out["iforest_return"] = np.where(base.eq("iforest"), vals, np.nan)
    return out


def _safe_baseline_thresholds(baseline_best: dict | None, df: pd.DataFrame | None = None) -> tuple[float, float]:
    baseline_best = baseline_best or {}
    z = _num(baseline_best.get("z_threshold"))
    i = _num(baseline_best.get("iforest_rate_pct"))
    # Also support formatted JSON rows.
    rows = baseline_best.get("rows") if isinstance(baseline_best, dict) else None
    if isinstance(rows, list):
        for row in rows:
            b = str(row.get("baseline", "")).lower()
            val = _num(row.get("z_threshold_or_iforest_rate_pct"))
            if b == "zscore" and np.isfinite(val):
                z = val
            if b == "iforest" and np.isfinite(val):
                i = val
    if (not np.isfinite(z) or not np.isfinite(i)) and df is not None and not df.empty:
        bdf = normalise_baseline_search_frame(df)
        if not np.isfinite(z) and "z_threshold" in bdf.columns:
            vals = pd.to_numeric(bdf.loc[bdf.get("baseline", "").astype(str).str.lower().eq("zscore"), "z_threshold"], errors="coerce").dropna()
            if not vals.empty:
                z = float(vals.iloc[0])
        if not np.isfinite(i) and "iforest_rate_pct" in bdf.columns:
            vals = pd.to_numeric(bdf.loc[bdf.get("baseline", "").astype(str).str.lower().eq("iforest"), "iforest_rate_pct"], errors="coerce").dropna()
            if not vals.empty:
                i = float(vals.iloc[0])
    return z, i


def format_ml_search_results(df: pd.DataFrame, mode: str | None = None, baseline_best: dict | None = None, baseline_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    raw = df.copy()
    if "stage_candidate" in raw.columns and "candidate" not in raw.columns:
        raw["candidate"] = raw["stage_candidate"]
    z_thr, if_rate = _safe_baseline_thresholds(baseline_best, baseline_df)
    if np.isfinite(z_thr):
        raw["z_threshold"] = z_thr
    if np.isfinite(if_rate):
        raw["iforest_rate_pct"] = if_rate
    if "severe_warning_rate" not in raw.columns:
        raw["severe_warning_rate"] = raw.get("severe_signal_rate", np.nan)
    if "severe_exposure" not in raw.columns and "crash_exposure" in raw.columns:
        raw["severe_exposure"] = raw["crash_exposure"]
    if "crash_exposure" not in raw.columns and "severe_exposure" in raw.columns:
        raw["crash_exposure"] = raw["severe_exposure"]
    if "high_medium_signal_rate" not in raw.columns and "medium_signal_rate" in raw.columns:
        raw["high_medium_signal_rate"] = raw["medium_signal_rate"]
    if "medium_signal_rate" not in raw.columns and "high_medium_signal_rate" in raw.columns:
        raw["medium_signal_rate"] = raw["high_medium_signal_rate"]

    mode_val = mode or (str(raw["prediction_mode"].dropna().iloc[0]) if "prediction_mode" in raw.columns and raw["prediction_mode"].notna().any() else "")
    is_holdout = "holdout" in str(mode_val).lower()
    mode_cols = ["train_fraction"] if is_holdout else ["initial_train_size", "retrain_every"]
    input_cols = [
        "ticker", "period", "prediction_mode",
        *BLANK_B,
        *mode_cols,
        "window", "forward_horizon", "risk_threshold",
        "low_medium_signal_rate", "high_medium_signal_rate", "high_signal_rate", "severe_signal_rate",
        "max_depth", "min_samples_leaf", "max_features", "n_estimators", "learning_rate",
        "reg_alpha", "reg_lambda", "subsample", "tcn_seq_len", "tcn_hidden", "tcn_epochs", "tcn_weight",
        "low_exposure", "low_medium_exposure", "high_medium_exposure", "high_exposure", "severe_exposure",
        "cooldown_days", "z_threshold", "iforest_rate_pct",
    ]
    result_cols = [
        "roc_auc", "recall", "precision", "severe_warning_rate",
        "ml_overlay_return", "buy_hold_return", "zscore_return", "iforest_return",
    ]

    rows = []
    for idx, r in raw.iterrows():
        roc = r.get("roc_auc", np.nan)
        rec = r.get("recall", np.nan)
        pre = r.get("precision", np.nan)
        ml = _num(r.get("ml_overlay_return"))
        bh = _num(r.get("buy_hold_return"))
        zr = _num(r.get("zscore_return"))
        ir = _num(r.get("iforest_return"))
        flags = _metric_flags(roc, rec, pre)
        ml_beats_bh = bool(np.isfinite(ml) and np.isfinite(bh) and ml > bh)
        ml_beats_z = bool(np.isfinite(ml) and np.isfinite(zr) and ml > zr)
        ml_beats_i = bool(np.isfinite(ml) and np.isfinite(ir) and ml > ir)
        row = {
            "overall_candidate": int(_num(r.get("overall_candidate", idx + 1), idx + 1)),
            "stage": r.get("stage", ""),
            "stage_candidate": int(_num(r.get("candidate", r.get("stage_candidate", idx + 1)), idx + 1)),
            **_blank_dict(BLANK_A),
        }
        for c in input_cols:
            if c in raw.columns:
                row[c] = r.get(c)
        row.update(_blank_dict(BLANK_D))
        for c in result_cols:
            row[c] = r.get(c, np.nan)
        row.update(_blank_dict(BLANK_E))
        # Download/presentation outputs intentionally omit the intermediate
        # classification/all-constraint flag columns. The optimisation logic can
        # still use raw in-memory search outputs before this formatter is applied.
        if "error" in raw.columns:
            row["error"] = r.get("error")
        rows.append(row)
    return pd.DataFrame(rows)


def format_ml_best_payload(payload: dict, baseline_best: dict | None = None) -> dict:
    if not payload:
        return {}
    df = pd.DataFrame([payload])
    mode = payload.get("prediction_mode")
    formatted = format_ml_search_results(df, mode=mode, baseline_best=baseline_best)
    return formatted.iloc[0].where(pd.notna(formatted.iloc[0]), None).to_dict() if not formatted.empty else dict(payload)


def _clean_json_value(obj: Any) -> Any:
    remove = {
        "candidate", "total_candidate", "total_candidates", "total_zscore_candidates", "total_iforest_candidates",
        "min_roc_auc_required", "min_recall_required", "min_precision_required", "min_f2_required",
        "high_label_threshold", "medium_label_threshold", "low_medium_label_threshold",
        "zscore_best", "iforest_best",
        "ml_beats_buy_hold", "ml_beats_zscore", "ml_beats_iforest", "ml_beats_all_baselines",
        "classification_requirement_met_1", "classification_requirement_met_2", "classification_requirement_met_3",
        "classification_requirement_met_4", "classification_requirement_met_5", "classification_requirement_met_6",
        "all_constraints_met_1", "all_constraints_met_2", "all_constraints_met_3",
        "all_constraints_met_4", "all_constraints_met_5", "all_constraints_met_6",
    }
    if isinstance(obj, dict):
        return {k: _clean_json_value(v) for k, v in obj.items() if k not in remove}
    if isinstance(obj, list):
        return [_clean_json_value(v) for v in obj]
    return obj


def format_baseline_best_payload(payload: dict) -> dict:
    if not payload:
        return {}
    rows = []
    for key, baseline in [("zscore_best", "zscore"), ("iforest_best", "iforest")]:
        src = payload.get(key)
        if isinstance(src, dict):
            row = dict(src)
            row["baseline"] = baseline
            row.setdefault("buy_hold_return", payload.get("buy_hold_return"))
            if baseline == "zscore":
                row.setdefault("z_threshold", payload.get("z_threshold"))
                row.setdefault("zscore_return", payload.get("zscore_return"))
            else:
                row.setdefault("iforest_rate_pct", payload.get("iforest_rate_pct"))
                row.setdefault("iforest_return", payload.get("iforest_return"))
            rows.append(row)
    if not rows:
        # Already flattened/formatted: preserve top-level and any row records.
        rows = payload.get("rows", []) if isinstance(payload.get("rows"), list) else []
    formatted_rows = []
    if rows:
        f = format_baseline_search_results(pd.DataFrame(rows))
        formatted_rows = [r for r in f.where(pd.notna(f), None).to_dict(orient="records")]

    z = payload.get("z_threshold")
    i = payload.get("iforest_rate_pct")
    for r in formatted_rows:
        b = str(r.get("baseline", "")).lower()
        val = r.get("z_threshold_or_iforest_rate_pct")
        if b == "zscore" and val is not None:
            z = val
        if b == "iforest" and val is not None:
            i = val
    out = {
        "ticker": payload.get("ticker"),
        "period": payload.get("period"),
        "prediction_mode": payload.get("prediction_mode"),
        "z_threshold": z,
        "iforest_rate_pct": i,
        "buy_hold_return": payload.get("buy_hold_return"),
        "zscore_return": payload.get("zscore_return"),
        "iforest_return": payload.get("iforest_return"),
        "rows": formatted_rows,
    }
    return _clean_json_value({k: v for k, v in out.items() if v is not None})


def write_optimisation_excel(df: pd.DataFrame, target, sheet_name: str = "OptimisationResults") -> None:
    """Write an optimisation DataFrame with wrapped cells and bounded widths."""
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.book[sheet_name]
        from openpyxl.styles import Alignment
        for col_cells in ws.columns:
            header = str(col_cells[0].value or "")
            if header.strip() == "":
                width = 3
            else:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                width = min(max(max_len + 2, 10), 24)
            ws.column_dimensions[col_cells[0].column_letter].width = width
            for cell in col_cells:
                cell.alignment = Alignment(wrap_text=True, shrink_to_fit=True, vertical="top")
