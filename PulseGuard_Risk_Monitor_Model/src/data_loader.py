from dataclasses import dataclass
from src.determinism import configure_determinism
configure_determinism()
from pathlib import Path
import pandas as pd

FROZEN_END_DATE = pd.Timestamp("2026-04-30")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "data_snapshots" / "2026-04-30"


@dataclass
class DatasetResult:
    frame: pd.DataFrame
    source_name: str
    source_url: str


def _normalise_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower().replace(" ", "_") for col in df.columns]
    else:
        df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    return df


def _snapshot_file(ticker: str, period: str) -> Path:
    safe_ticker = str(ticker).upper().replace("/", "-").replace(".", "-")
    return SNAPSHOT_DIR / f"{safe_ticker}_{period}_ohlcv.csv.gz"


def load_market_data(ticker: str = "AAPL", period: str = "5y") -> DatasetResult:
    """Load deterministic OHLCV snapshots only.

    Runtime Yahoo/yfinance fetching is intentionally disabled. Generate snapshots
    with `python scripts/prefreeze_ohlcv_snapshots.py` before running/deploying.
    """
    ticker = str(ticker).strip().upper()
    path = _snapshot_file(ticker, period)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen OHLCV snapshot for {ticker} {period}. "
            f"Expected: {path.as_posix()}. "
            "Run `python scripts/prefreeze_ohlcv_snapshots.py` from the project root, then restart the app."
        )

    df = pd.read_csv(path, compression="gzip")
    if df.empty:
        raise ValueError(f"Frozen OHLCV snapshot is empty for {ticker} {period}.")

    df = _normalise_yfinance_columns(df)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Frozen OHLCV snapshot for {ticker} {period} is missing columns: {missing}")

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
          .sort_values("timestamp", kind="mergesort")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )
    df = df[df["timestamp"] <= FROZEN_END_DATE].reset_index(drop=True)
    df["target"] = df["close"]
    df = df[["timestamp", "open", "high", "low", "close", "volume", "target"]]

    if len(df) < 180:
        raise ValueError("Not enough rows. Use at least 1y of data for this model.")

    return DatasetResult(
        frame=df,
        source_name=f"Frozen OHLCV snapshot: {ticker.upper()} ({period}, 1d, cutoff 2026-04-30)",
        source_url="local:data_snapshots/2026-04-30",
    )
