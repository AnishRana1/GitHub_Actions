"""Central deterministic runtime setup for PulseGuard.

This module is intentionally imported early by app, scripts and modelling code.
It cannot retroactively change interpreter-start settings such as PYTHONHASHSEED,
but it still documents and applies every deterministic control available at
runtime. GitHub Actions also sets these variables before Python starts.
"""
from __future__ import annotations

import os
import random
from typing import Any

GLOBAL_RANDOM_SEED = int(os.environ.get("PULSEGUARD_SEED", "42"))

_DETERMINISTIC_ENV = {
    "PYTHONHASHSEED": str(GLOBAL_RANDOM_SEED),
    "PULSEGUARD_SEED": str(GLOBAL_RANDOM_SEED),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XGBOOST_NUM_THREADS": "1",
    "LIGHTGBM_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def configure_determinism(seed: int = GLOBAL_RANDOM_SEED) -> int:
    for key, value in _DETERMINISTIC_ENV.items():
        os.environ.setdefault(key, str(value))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        if os.environ.get("PULSEGUARD_ENABLE_TORCH", "0") != "1":
            return seed
        import torch
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed


def reset_random_seeds(seed: int = GLOBAL_RANDOM_SEED) -> int:
    return configure_determinism(seed)


def deterministic_rng(seed: int = GLOBAL_RANDOM_SEED):
    import numpy as np
    return np.random.default_rng(seed)


def stable_json_payload(value: Any) -> str:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


configure_determinism(GLOBAL_RANDOM_SEED)
