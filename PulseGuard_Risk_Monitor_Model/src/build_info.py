"""Deterministic build identity helpers for PulseGuard.

The build id is not used for modelling. It is a traceability/audit value derived
from source/config/data-manifest files that should be identical for identical
deployable builds.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from src.data_loader import FROZEN_END_DATE
from src.features import FEATURE_SCHEMA_VERSION
from src.pipeline_cache import PIPELINE_CACHE_VERSION
from src.optimisation_cache import CACHE_VERSION as OPTIMISATION_CACHE_VERSION
from src.determinism import GLOBAL_RANDOM_SEED

BUILD_ID_VERSION = "pulseguard-build-id-v1"

_INCLUDE_GLOBS = (
    "app/*.py",
    "src/*.py",
    "scripts/*.py",
    "requirements.txt",
    "runtime.txt",
    "supported_tickers.json",
    "data_snapshots/2026-04-30/ohlcv_snapshot_manifest.json",
    "optimisation_results/**/*.json",
    "optimisation_results/**/*.csv",
)

_EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_build_files(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in _INCLUDE_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in _EXCLUDE_PARTS for part in rel.parts):
                continue
            if path.suffix in _EXCLUDE_SUFFIXES:
                continue
            if rel in seen:
                continue
            seen.add(rel)
            yield rel


@lru_cache(maxsize=1)
def get_build_info() -> dict:
    root = project_root()
    digest = hashlib.sha256()
    files = sorted(_iter_build_files(root), key=lambda p: p.as_posix())
    for rel in files:
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / rel).read_bytes())
        digest.update(b"\0")

    full_hash = digest.hexdigest()
    payload = {
        "build_id_version": BUILD_ID_VERSION,
        "build_id": full_hash[:16],
        "build_hash_sha256": full_hash,
        "file_count": len(files),
        "frozen_data_cutoff": str(FROZEN_END_DATE.date()),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "pipeline_cache_version": PIPELINE_CACHE_VERSION,
        "optimisation_cache_version": OPTIMISATION_CACHE_VERSION,
        "global_random_seed": GLOBAL_RANDOM_SEED,
    }
    return payload


def write_build_info(path: str | Path | None = None) -> dict:
    info = get_build_info()
    target = Path(path) if path else project_root() / "deterministic_build_info.json"
    target.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


if __name__ == "__main__":
    print(json.dumps(write_build_info(), indent=2, sort_keys=True))
