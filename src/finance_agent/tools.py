"""Tool functions exposed to the finance agent.

Each function has full type hints and a Google-style docstring so that the
ADK can auto-generate the JSON schema the LLM uses when deciding to call it.
"""

from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd

from finance_agent.data import default_month, filter_month, load_csv


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def load_transactions(path: str | None = None) -> dict:
    """Load transactions from a CSV file and return a summary.

    Args:
        path: Optional path to the CSV file. Defaults to the FINANCE_CSV_PATH
              environment variable or the bundled sample file.

    Returns:
        A dict with keys ``rows`` (int), ``date_min`` (str), ``date_max`` (str),
        and ``accounts`` (list[str]).
    """
    df = load_csv(path)
    return {
        "rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "accounts": sorted(df["account"].unique().tolist()),
    }


def spending_by_category(
    month: str | None = None,
    account: str | None = None,
    category: str | None = None,
) -> dict[str, float]:
    """Return total spending per category for a given month.

    Spending rows have a negative ``amount``; the returned values are
    positive totals (absolute values).

    Args:
        month: Month in ``YYYY-MM`` format. Defaults to the most recent
               complete calendar month.
        account: Optional bank account filter (``"Checking"`` or
                 ``"Credit Card"``). Case-insensitive.
        category: Optional single category to return (e.g. ``"Groceries"``).
                  Returns a single-key dict. Case-insensitive.

    Returns:
        Dict mapping category name to total spend (positive float, USD).
        Returns an empty dict if no spending data exists for that month.
    """
    df = load_csv()
    resolved_month = month or default_month(df)
    df = filter_month(df, resolved_month)

    if account and str(account).lower() not in ("null", "none", ""):
        df = df[df["account"].str.lower() == account.lower()]

    if category and str(category).lower() not in ("null", "none", ""):
        df = df[df["category"].str.lower() == category.lower()]

    spend = df[df["amount"] < 0].copy()
    if spend.empty:
        return {}

    result = spend.groupby("category")["amount"].sum().abs().round(2)
    return {k: float(v) for k, v in result.items()}


def top_merchants(
    month: str | None = None,
    n: int = 5,
) -> list[dict]:
    """Return the top merchants by total spending for a given month.

    Args:
        month: Month in ``YYYY-MM`` format. Defaults to the most recent
               complete calendar month.  Pass ``"all"`` to aggregate across
               all months in the data set.
        n: Number of merchants to return (default 5).

    Returns:
        List of dicts, each with keys ``merchant``, ``total_spend`` (USD),
        and ``transaction_count``.  Sorted descending by ``total_spend``.
    """
    df = load_csv()
    if month and month.lower() != "all":
        df = filter_month(df, month)
    elif not month:
        resolved = default_month(df)
        df = filter_month(df, resolved)

    spend = df[df["amount"] < 0].copy()
    if spend.empty:
        return []

    grouped = (
        spend.groupby("merchant")["amount"]
        .agg(total_spend="sum", transaction_count="count")
        .reset_index()
    )
    grouped["total_spend"] = grouped["total_spend"].abs().round(2)
    top = grouped.nlargest(n, "total_spend")

    return top.to_dict(orient="records")


def monthly_summary(month: str | None = None) -> dict:
    """Return an income / spend / net summary for a given month.

    Args:
        month: Month in ``YYYY-MM`` format. Defaults to the most recent
               complete calendar month.

    Returns:
        Dict with keys ``month`` (str), ``total_income`` (float),
        ``total_spend`` (float), ``net`` (float), and
        ``top_categories`` (list[dict] with ``category`` and ``spend``).
    """
    df = load_csv()
    resolved_month = month or default_month(df)
    df = filter_month(df, resolved_month)

    if df.empty:
        return {
            "month": resolved_month,
            "total_income": 0.0,
            "total_spend": 0.0,
            "net": 0.0,
            "top_categories": [],
        }

    income = round(float(df[df["amount"] > 0]["amount"].sum()), 2)
    spend = round(abs(float(df[df["amount"] < 0]["amount"].sum())), 2)

    cat_spend = (
        df[df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .abs()
        .nlargest(3)
        .reset_index()
    )
    top_cats = [
        {"category": row["category"], "spend": round(float(row["amount"]), 2)}
        for _, row in cat_spend.iterrows()
    ]

    return {
        "month": resolved_month,
        "total_income": income,
        "total_spend": spend,
        "net": round(income - spend, 2),
        "top_categories": top_cats,
    }


def recurring_subscriptions() -> list[dict]:
    """Detect recurring subscriptions using a monthly-cadence heuristic.

    A merchant is considered a subscription when it appears in at least
    3 of the last 6 months with amounts within 5 % of each other.

    Returns:
        List of dicts with keys ``merchant``, ``avg_amount`` (USD),
        ``months_seen`` (int), and ``last_charge`` (str, YYYY-MM-DD).
        Sorted by ``avg_amount`` descending.
    """
    df = load_csv()
    spend = df[df["amount"] < 0].copy()

    today = pd.Timestamp.today()
    cutoff = today - pd.DateOffset(months=6)
    spend = spend[spend["date"] >= cutoff]

    spend["month"] = spend["date"].dt.to_period("M")

    results: list[dict] = []

    for merchant, group in spend.groupby("merchant"):
        months = group["month"].unique()
        if len(months) < 3:
            continue

        amounts = group["amount"].abs()
        mean_amt = amounts.mean()
        if mean_amt == 0:
            continue

        # Within 5% variance
        if (amounts.std() / mean_amt) > 0.05:
            continue

        results.append(
            {
                "merchant": merchant,
                "avg_amount": round(float(mean_amt), 2),
                "months_seen": int(len(months)),
                "last_charge": str(group["date"].max().date()),
            }
        )

    return sorted(results, key=lambda x: x["avg_amount"], reverse=True)


def budget_check(
    category: str,
    monthly_limit: float,
    month: str | None = None,
) -> dict:
    """Check whether spending in a category exceeds a monthly budget.

    Args:
        category: The spending category (e.g. ``"Dining"``). Case-insensitive.
        monthly_limit: The budget limit in USD.
        month: Month in ``YYYY-MM`` format. Defaults to the most recent
               complete calendar month.

    Returns:
        Dict with keys ``category``, ``month``, ``monthly_limit``,
        ``actual_spend``, ``over_budget`` (bool), and ``difference``
        (positive means over-budget, negative means under).
    """
    df = load_csv()
    resolved_month = month or default_month(df)

    by_cat = spending_by_category(month=resolved_month)

    # Case-insensitive lookup
    actual = 0.0
    for cat, amt in by_cat.items():
        if cat.lower() == category.lower():
            actual = amt
            break

    difference = round(actual - monthly_limit, 2)
    return {
        "category": category,
        "month": resolved_month,
        "monthly_limit": round(monthly_limit, 2),
        "actual_spend": round(actual, 2),
        "over_budget": actual > monthly_limit,
        "difference": difference,
    }


def search_transactions(
    query: str,
    month: str | None = None,
) -> list[dict]:
    """Search transactions by keyword in description or merchant name.

    Args:
        query: Case-insensitive substring to search for.
        month: Optional month filter in ``YYYY-MM`` format.  If omitted,
               searches the entire history.

    Returns:
        List of dicts with keys ``date``, ``description``, ``merchant``,
        ``category``, ``amount``, ``account``.  At most 50 results.
    """
    df = load_csv()
    if month:
        df = filter_month(df, month)

    q = query.lower()
    mask = df["description"].str.lower().str.contains(q, na=False) | \
           df["merchant"].str.lower().str.contains(q, na=False)
    matches = df[mask].head(50)

    records = matches.copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    return records.to_dict(orient="records")
