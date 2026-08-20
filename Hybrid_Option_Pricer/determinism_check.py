"""Deterministic-behaviour measurement for CI.

This script trains/measures the default deployed model from the embedded snapshot
without executing the Streamlit UI and without calling live market-data APIs.

Usage:
  python determinism_check.py
      Run two measurements in the same process and compare them.

  python determinism_check.py --json
      Emit one stable JSON measurement. GitHub Actions runs this twice in two
      separate Python processes and diffs the two JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


MEASUREMENT_DECIMALS = 10


def _load_app_core() -> dict[str, Any]:
    app_path = Path(__file__).with_name("app.py")
    source = app_path.read_text()
    marker = "# ===== UI ====="
    if marker not in source:
        raise RuntimeError("Could not find app UI marker; refusing to execute full Streamlit app.")
    namespace = {"__file__": str(app_path), "__name__": "app_core_for_determinism_check"}
    exec(compile(source.split(marker, 1)[0], str(app_path), "exec"), namespace)
    return namespace


def _round_float(value: Any) -> float:
    return round(float(value), MEASUREMENT_DECIMALS)


def _measurement(ns: dict[str, Any]) -> dict[str, Any]:
    context = ns["get_default_model_context"]()
    training_data, _selected_data, _selected_spot, loaded_tickers = ns["load_training_data"](
        context["ticker"], context["option_type"], context["training_mode"]
    )
    bundle, train_df, test_df, metrics = ns["train_model"](
        training_data,
        context["risk_free_rate"],
        context["train_fraction"],
        context["option_type"],
        context["ticker"],
        context["model_params"],
        xgb_progress_callback=None,
    )
    preds = ns["hybrid_predict"](
        bundle,
        test_df["spot_price"].head(25).values,
        test_df["strike_price"].head(25).values,
        test_df["time_to_maturity"].head(25).values,
        context["risk_free_rate"],
        test_df["market_volatility"].head(25).values,
        context["option_type"],
    )
    return {
        "cache_key": ns["cache_key_for"](
            context["ticker"],
            context["option_type"],
            context["training_mode"],
            context["risk_free_rate"],
            context["train_fraction"],
            context["model_params"],
        ),
        "snapshot_identifier": ns["snapshot_identifier"](),
        "loaded_tickers": sorted(str(t) for t in loaded_tickers),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": {
            str(k): _round_float(v)
            for k, v in sorted(metrics.items())
            if isinstance(v, (int, float, np.floating))
        },
        "prediction_probe": [_round_float(x) for x in np.asarray(preds).ravel().tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure deterministic default-model behaviour.")
    parser.add_argument("--json", action="store_true", help="emit one stable JSON measurement")
    args = parser.parse_args()

    ns = _load_app_core()
    first = _measurement(ns)

    if args.json:
        print(json.dumps(first, sort_keys=True, separators=(",", ":")))
        return

    second = _measurement(ns)
    print(json.dumps({"first": first, "second": second}, indent=2, sort_keys=True, default=str))
    if first != second:
        raise SystemExit("Determinism check failed: repeated measurements differ.")
    print("Determinism check passed: repeated measurements are identical.")


if __name__ == "__main__":
    main()
