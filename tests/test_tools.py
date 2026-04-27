"""Unit tests for finance_agent/tools.py.

All tests use a small in-memory fixture DataFrame; no CSV file or network
calls are made.
"""

from __future__ import annotations

import pandas as pd
import pytest

import finance_agent.data as data_module
from finance_agent.tools import (
    budget_check,
    load_transactions,
    monthly_summary,
    recurring_subscriptions,
    search_transactions,
    spending_by_category,
    top_merchants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure the module-level DataFrame cache is cleared between tests."""
    data_module.reset_cache()
    yield
    data_module.reset_cache()


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    """Small but realistic transaction fixture spanning Jan–Mar 2026."""
    rows = [
        # January 2026
        ("2026-01-01", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-01-05", "Whole Foods", "Whole Foods", "Groceries", -90.00, "Credit Card"),
        ("2026-01-10", "Netflix", "Netflix", "Subscriptions", -15.99, "Credit Card"),
        ("2026-01-12", "Shell Gas", "Shell", "Transport", -50.00, "Credit Card"),
        ("2026-01-15", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-01-20", "Chipotle", "Chipotle", "Dining", -18.00, "Credit Card"),
        ("2026-01-25", "Spotify", "Spotify", "Subscriptions", -9.99, "Credit Card"),
        ("2026-01-28", "Kroger", "Kroger", "Groceries", -75.00, "Credit Card"),
        # February 2026
        ("2026-02-01", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-02-05", "Whole Foods", "Whole Foods", "Groceries", -85.00, "Credit Card"),
        ("2026-02-10", "Netflix", "Netflix", "Subscriptions", -15.99, "Credit Card"),
        ("2026-02-14", "Valentine Dinner", "Le Coucou", "Dining", -200.00, "Credit Card"),
        ("2026-02-15", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-02-20", "Spotify", "Spotify", "Subscriptions", -9.99, "Credit Card"),
        ("2026-02-22", "Shell Gas", "Shell", "Transport", -48.00, "Credit Card"),
        ("2026-02-26", "Amazon", "Amazon", "Shopping", -60.00, "Credit Card"),
        # March 2026
        ("2026-03-01", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-03-05", "Whole Foods", "Whole Foods", "Groceries", -95.00, "Credit Card"),
        ("2026-03-10", "Netflix", "Netflix", "Subscriptions", -15.99, "Credit Card"),
        ("2026-03-12", "Shell Gas", "Shell", "Transport", -52.00, "Credit Card"),
        ("2026-03-15", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-03-18", "Chipotle", "Chipotle", "Dining", -22.00, "Credit Card"),
        ("2026-03-22", "Spotify", "Spotify", "Subscriptions", -9.99, "Credit Card"),
        ("2026-03-28", "Kroger", "Kroger", "Groceries", -80.00, "Credit Card"),
    ]
    df = pd.DataFrame(rows, columns=["date", "description", "merchant", "category", "amount", "account"])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df


@pytest.fixture
def patch_csv(fixture_df, monkeypatch):
    """Patch load_csv to return the fixture DataFrame without hitting disk."""
    monkeypatch.setattr(data_module, "_cache", fixture_df)
    return fixture_df


# ---------------------------------------------------------------------------
# load_transactions
# ---------------------------------------------------------------------------


def test_load_transactions_returns_summary(patch_csv, tmp_path):
    result = load_transactions()
    assert result["rows"] == len(patch_csv)
    assert result["date_min"] == "2026-01-01"
    assert result["date_max"] == "2026-03-28"
    assert "Checking" in result["accounts"]
    assert "Credit Card" in result["accounts"]


# ---------------------------------------------------------------------------
# spending_by_category
# ---------------------------------------------------------------------------


def test_spending_by_category_specific_month(patch_csv):
    result = spending_by_category(month="2026-01")
    assert "Groceries" in result
    assert abs(result["Groceries"] - 165.00) < 0.01  # 90 + 75
    assert "Income" not in result  # income rows excluded


def test_spending_by_category_defaults_to_last_full_month(patch_csv):
    # No month arg → should pick a valid month from the fixture data and
    # return a non-empty dict (fixture spans Jan–Mar 2026)
    result = spending_by_category()
    assert isinstance(result, dict)
    # At least one spending category must exist
    assert len(result) > 0


def test_spending_by_category_no_data(patch_csv):
    result = spending_by_category(month="2020-01")
    assert result == {}


def test_spending_by_category_account_filter(patch_csv):
    result = spending_by_category(month="2026-01", account="Credit Card")
    assert "Groceries" in result
    # All Credit Card items for Jan: Whole Foods 90, Netflix 15.99, Shell 50,
    # Chipotle 18, Spotify 9.99, Kroger 75
    total = sum(result.values())
    assert abs(total - (90 + 15.99 + 50 + 18 + 9.99 + 75)) < 0.01


def test_spending_by_category_unknown_account(patch_csv):
    result = spending_by_category(month="2026-01", account="Savings")
    assert result == {}


# ---------------------------------------------------------------------------
# top_merchants
# ---------------------------------------------------------------------------


def test_top_merchants_returns_n(patch_csv):
    result = top_merchants(month="2026-03", n=3)
    assert len(result) <= 3
    assert all("merchant" in r and "total_spend" in r for r in result)


def test_top_merchants_sorted_descending(patch_csv):
    result = top_merchants(month="2026-03", n=5)
    spends = [r["total_spend"] for r in result]
    assert spends == sorted(spends, reverse=True)


def test_top_merchants_all_months(patch_csv):
    result = top_merchants(month="all", n=5)
    # Employer Inc is income (positive), should not appear
    merchants = [r["merchant"] for r in result]
    assert "Employer Inc" not in merchants


def test_top_merchants_no_data(patch_csv):
    result = top_merchants(month="2019-06")
    assert result == []


# ---------------------------------------------------------------------------
# monthly_summary
# ---------------------------------------------------------------------------


def test_monthly_summary_structure(patch_csv):
    result = monthly_summary(month="2026-01")
    assert result["month"] == "2026-01"
    assert result["total_income"] == 6000.00
    assert result["total_spend"] > 0
    assert result["net"] == round(result["total_income"] - result["total_spend"], 2)
    assert isinstance(result["top_categories"], list)
    assert len(result["top_categories"]) <= 3


def test_monthly_summary_empty_month(patch_csv):
    result = monthly_summary(month="2025-06")
    assert result["total_income"] == 0.0
    assert result["total_spend"] == 0.0
    assert result["net"] == 0.0
    assert result["top_categories"] == []


# ---------------------------------------------------------------------------
# recurring_subscriptions
# ---------------------------------------------------------------------------


def test_recurring_subscriptions_finds_netflix(patch_csv):
    subs = recurring_subscriptions()
    merchants = [s["merchant"] for s in subs]
    assert "Netflix" in merchants


def test_recurring_subscriptions_finds_spotify(patch_csv):
    subs = recurring_subscriptions()
    merchants = [s["merchant"] for s in subs]
    assert "Spotify" in merchants


def test_recurring_subscriptions_excludes_one_off(patch_csv):
    subs = recurring_subscriptions()
    merchants = [s["merchant"] for s in subs]
    # Le Coucou appeared only once
    assert "Le Coucou" not in merchants


def test_recurring_subscriptions_structure(patch_csv):
    subs = recurring_subscriptions()
    for s in subs:
        assert "merchant" in s
        assert "avg_amount" in s
        assert "months_seen" in s
        assert "last_charge" in s
        assert s["months_seen"] >= 3


# ---------------------------------------------------------------------------
# budget_check
# ---------------------------------------------------------------------------


def test_budget_check_under(patch_csv):
    result = budget_check("Groceries", 200.00, month="2026-01")
    assert result["actual_spend"] == 165.00
    assert result["over_budget"] is False
    assert result["difference"] < 0


def test_budget_check_over(patch_csv):
    result = budget_check("Dining", 50.00, month="2026-02")
    # February dining = Valentine dinner 200
    assert result["over_budget"] is True
    assert result["difference"] > 0


def test_budget_check_unknown_category(patch_csv):
    result = budget_check("Vacation", 500.00, month="2026-01")
    assert result["actual_spend"] == 0.0
    assert result["over_budget"] is False


def test_budget_check_case_insensitive(patch_csv):
    r1 = budget_check("groceries", 200.00, month="2026-01")
    r2 = budget_check("Groceries", 200.00, month="2026-01")
    assert r1["actual_spend"] == r2["actual_spend"]


# ---------------------------------------------------------------------------
# search_transactions
# ---------------------------------------------------------------------------


def test_search_transactions_finds_match(patch_csv):
    results = search_transactions("netflix")
    assert len(results) > 0
    assert all("netflix" in r["merchant"].lower() for r in results)


def test_search_transactions_month_filter(patch_csv):
    results = search_transactions("whole foods", month="2026-01")
    assert len(results) == 1
    assert results[0]["date"] == "2026-01-05"


def test_search_transactions_no_match(patch_csv):
    results = search_transactions("zzznomatch")
    assert results == []


def test_search_transactions_case_insensitive(patch_csv):
    r1 = search_transactions("NETFLIX")
    r2 = search_transactions("netflix")
    assert len(r1) == len(r2)
