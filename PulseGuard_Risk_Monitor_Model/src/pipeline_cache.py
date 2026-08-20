"""
Permanent model pipeline cache – results never expire automatically.
Only a "Retrain" button action replaces a specific entry for a given input set.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.data_loader import FROZEN_END_DATE
from src.features import FEATURE_SCHEMA_VERSION

PIPELINE_CACHE_VERSION = "pulseguard-model-pipeline-cache-v5-deterministic"
PIPELINE_CACHE_DIR = Path(__file__).resolve().parents[1] / ".pulseguard_model_cache"
PIPELINE_INDEX_FILE = PIPELINE_CACHE_DIR / "index.json"


def today_key(now: Optional[datetime] = None) -> str:
    """Kept for backward-compat; no longer used as cache expiry key."""
    return (now or datetime.now()).strftime("%Y-%m-%d")


def _normalise_value(value: Any) -> Any:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Cache inputs must be finite numbers.")
        return round(value, 10)
    if isinstance(value, int) or isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        # Treat common ticker-like fields consistently without changing semantic labels.
        return stripped.upper() if stripped and stripped.replace("-", "").replace("_", "").isalnum() else stripped
    if isinstance(value, dict):
        return {str(k): _normalise_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise_value(v) for v in value]
    return str(value)


def _strip_volatile_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    volatile = {"cache_date", "created_at", "updated_at", "generated_at", "timestamp", "last_full_opt_time", "last_full_opt_display_time", "run_started_at", "run_finished_at"}
    return {k: v for k, v in dict(metadata or {}).items() if k not in volatile}


def make_pipeline_cache_key(**inputs: Any) -> str:
    """Keyed only by effective modelling inputs. No date component → permanent."""
    # Strip cache_date so the key doesn't change day-to-day for the same inputs
    inputs_clean = _strip_volatile_metadata(inputs)
    payload = {
        "version": PIPELINE_CACHE_VERSION,
        "frozen_data_cutoff": str(FROZEN_END_DATE.date()),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "inputs": _normalise_value(inputs_clean),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _empty_index() -> Dict[str, Any]:
    return {"version": PIPELINE_CACHE_VERSION, "items": {}}


def _load_index() -> Dict[str, Any]:
    if not PIPELINE_INDEX_FILE.exists():
        return _empty_index()
    try:
        index = json.loads(PIPELINE_INDEX_FILE.read_text())
    except Exception:
        return _empty_index()
    # Only invalidate on schema version change, NOT on date change
    if index.get("version") != PIPELINE_CACHE_VERSION:
        clear_pipeline_cache()
        return _empty_index()
    if not isinstance(index.get("items"), dict):
        return _empty_index()
    return index


def _save_index(index: Dict[str, Any]) -> None:
    PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index["version"] = PIPELINE_CACHE_VERSION
    index.setdefault("items", {})
    PIPELINE_INDEX_FILE.write_text(json.dumps(index, indent=2, sort_keys=True))


def get_pipeline_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    index = _load_index()
    item = index.get("items", {}).get(cache_key)
    if not item:
        return None
    path = PIPELINE_CACHE_DIR / item.get("file", "")
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def get_pipeline_cache_metadata(cache_key: str) -> Optional[Dict[str, Any]]:
    """Return the index entry for a cache key (created_at, compute_duration_s, metadata)."""
    index = _load_index()
    item = index.get("items", {}).get(cache_key)
    return dict(item) if item else None


def store_pipeline_cache(
    cache_key: str, result: Dict[str, Any], metadata: Dict[str, Any],
    compute_duration_s: Optional[float] = None,
) -> None:
    PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{cache_key}.pkl"
    path = PIPELINE_CACHE_DIR / filename
    with path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    index = _load_index()
    entry: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "file": filename,
        "metadata": _strip_volatile_metadata(metadata),
    }
    if compute_duration_s is not None and compute_duration_s > 0:
        entry["compute_duration_s"] = float(compute_duration_s)
    index.setdefault("items", {})[cache_key] = entry
    _save_index(index)


def invalidate_pipeline_cache(cache_key: str) -> None:
    """Remove a single entry from the cache (used by 'Retrain' button)."""
    index = _load_index()
    item = index.get("items", {}).pop(cache_key, None)
    if item:
        path = PIPELINE_CACHE_DIR / item.get("file", "")
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        _save_index(index)


def clear_pipeline_cache() -> None:
    if PIPELINE_CACHE_DIR.exists():
        for path in PIPELINE_CACHE_DIR.glob("*.pkl"):
            try:
                path.unlink()
            except Exception:
                pass
    _save_index(_empty_index())
