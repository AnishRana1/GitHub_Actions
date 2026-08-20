"""
Permanent optimisation cache – results never expire automatically.
Only the "Retrain selected optimisations" button replaces a specific entry.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.data_loader import FROZEN_END_DATE
from src.features import FEATURE_SCHEMA_VERSION

CACHE_VERSION = "pulseguard-optimisation-cache-v5-deterministic"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".pulseguard_optimisation_cache"
CACHE_FILE = CACHE_DIR / "optimised_parameters.json"


def today_key(now: Optional[datetime] = None) -> str:
    """Return the local calendar date (kept for backward-compat with imports)."""
    return (now or datetime.now()).strftime("%Y-%m-%d")


def make_cache_key(
    *,
    ticker: str,
    period: str,
    prediction_mode: str,
    optimisation_tickers=None,
) -> str:
    if isinstance(optimisation_tickers, str):
        universe = list(dict.fromkeys([x.strip().upper() for x in optimisation_tickers.split(",") if x.strip()]))
    elif optimisation_tickers:
        universe = list(dict.fromkeys([str(x).strip().upper() for x in optimisation_tickers if str(x).strip()]))
    else:
        universe = []
    payload = {
        "version": CACHE_VERSION,
        "ticker": ticker.strip().upper(),
        "period": period,
        "prediction_mode": prediction_mode,
        "optimisation_tickers": universe,
        "frozen_data_cutoff": str(FROZEN_END_DATE.date()),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _empty_cache() -> Dict[str, Any]:
    return {"version": CACHE_VERSION, "items": {}}


def load_cache() -> Dict[str, Any]:
    """Load the persistent cache. Only resets if the schema version changes."""
    if not CACHE_FILE.exists():
        return _empty_cache()
    try:
        cache = json.loads(CACHE_FILE.read_text())
    except Exception:
        return _empty_cache()
    # Only invalidate on schema version change, NOT on date change
    if cache.get("version") != CACHE_VERSION:
        return _empty_cache()
    if not isinstance(cache.get("items"), dict):
        return _empty_cache()
    return cache


def save_cache(cache: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache["version"] = CACHE_VERSION
    cache.setdefault("items", {})
    CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_cached_result(cache_key: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    cache = load_cache()
    item = cache.get("items", {}).get(cache_key)
    if not item:
        return None
    return item.get("raw_opt", {}), item.get("report", {})



def _strip_volatile_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    volatile = {"cache_date", "created_at", "updated_at", "generated_at", "timestamp", "last_full_opt_time", "last_full_opt_display_time", "run_started_at", "run_finished_at"}
    return {k: v for k, v in dict(metadata or {}).items() if k not in volatile}

def store_cached_result(
    cache_key: str,
    raw_opt: Dict[str, Any],
    report: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    cache = load_cache()
    cache.setdefault("items", {})[cache_key] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": _strip_volatile_metadata(metadata),
        "raw_opt": raw_opt,
        "report": report,
    }
    save_cache(cache)


def clear_today_cache() -> None:
    """Clear all cached entries (kept for backward-compat with scripts)."""
    save_cache(_empty_cache())
