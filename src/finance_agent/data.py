"""CSV loading and pandas helpers."""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path

import pandas as pd

_cache: pd.DataFrame | None = None

# Set per-request by the FastAPI layer when the user uploads a file.
_active_csv: ContextVar[str | None] = ContextVar("_active_csv", default=None)


def load_csv(path: str | None = None) -> pd.DataFrame:
    """Load and cache transactions from a CSV file.

    Args:
        path: Absolute or relative path to the CSV. Defaults to the
              FINANCE_CSV_PATH env var, then ./data/sample_transactions.csv.

    Returns:
        DataFrame with columns: date, description, merchant, category,
        amount, account.  The ``date`` column is parsed as datetime.
    """
    global _cache

    # Per-request uploaded file takes priority over everything else.
    if path is None:
        path = _active_csv.get(None)

    if _cache is not None and path is None:
        return _cache

    if path is None:
        path = os.getenv(
            "FINANCE_CSV_PATH",
            str(Path(__file__).parents[3] / "data" / "sample_transactions.csv"),
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"])

    if path == os.getenv("FINANCE_CSV_PATH", str(
        Path(__file__).parents[3] / "data" / "sample_transactions.csv"
    )):
        _cache = df

    return df


def reset_cache() -> None:
    """Clear the in-memory cache (useful for testing)."""
    global _cache
    _cache = None


def default_month(df: pd.DataFrame) -> str:
    """Return the most recent complete calendar month as 'YYYY-MM'.

    A month is 'complete' if it is strictly before the current month.
    Falls back to the latest month present in the data if the data is old.
    """
    today = pd.Timestamp.today().normalize()
    first_of_current = today.replace(day=1)
    last_full = first_of_current - pd.DateOffset(months=1)

    available = df["date"].dt.to_period("M").unique()
    candidate = pd.Period(last_full, "M")

    if candidate in available:
        return str(candidate)

    # Data is older — use the latest month in the file
    return str(max(available))


def filter_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Return rows whose date falls within *month* (format 'YYYY-MM')."""
    return df[df["date"].dt.to_period("M").astype(str) == month]
