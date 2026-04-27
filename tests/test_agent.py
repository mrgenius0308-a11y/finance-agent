"""Integration tests for the agent layer — Groq backend mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import finance_agent.data as data_module
from finance_agent import memory as mem_module


@pytest.fixture(autouse=True)
def reset_state():
    data_module.reset_cache()
    mem_module.reset_context()
    yield
    data_module.reset_cache()
    mem_module.reset_context()


@pytest.fixture
def fixture_df():
    rows = [
        ("2026-03-01", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-03-05", "Whole Foods", "Whole Foods", "Groceries", -95.00, "Credit Card"),
        ("2026-03-10", "Netflix", "Netflix", "Subscriptions", -15.99, "Credit Card"),
        ("2026-03-18", "Chipotle", "Chipotle", "Dining", -22.00, "Credit Card"),
        ("2026-03-22", "Spotify", "Spotify", "Subscriptions", -9.99, "Credit Card"),
        ("2026-03-28", "Kroger", "Kroger", "Groceries", -80.00, "Credit Card"),
        ("2026-03-15", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-02-05", "Whole Foods", "Whole Foods", "Groceries", -85.00, "Credit Card"),
        ("2026-02-10", "Netflix", "Netflix", "Subscriptions", -15.99, "Credit Card"),
        ("2026-02-01", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
        ("2026-02-15", "Paycheck", "Employer Inc", "Income", 3000.00, "Checking"),
    ]
    df = pd.DataFrame(rows, columns=["date", "description", "merchant", "category", "amount", "account"])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df


@pytest.fixture
def patch_csv(fixture_df, monkeypatch):
    monkeypatch.setattr(data_module, "_cache", fixture_df)
    return fixture_df


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

def test_tools_schema_has_all_tools():
    from finance_agent.agent import TOOLS_SCHEMA
    names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    expected = {"load_transactions", "spending_by_category", "top_merchants",
                "monthly_summary", "recurring_subscriptions", "budget_check",
                "search_transactions"}
    assert expected == names


def test_registry_matches_schema():
    from finance_agent.agent import TOOLS_SCHEMA, _REGISTRY
    for t in TOOLS_SCHEMA:
        assert t["function"]["name"] in _REGISTRY


def test_system_prompt_has_key_phrases():
    from finance_agent.agent import SYSTEM_PROMPT
    assert "tool" in SYSTEM_PROMPT.lower()
    assert "memory" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# run_turn — LLM mocked
# ---------------------------------------------------------------------------

def _mock_groq_response(text: str, tool_calls=None):
    """Build a mock Groq chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = tool_calls
    choice.finish_reason = "tool_calls" if tool_calls else "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_run_turn_returns_final_text(patch_csv, monkeypatch):
    final_resp = _mock_groq_response("You spent $175.00 on Groceries.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = final_resp

    with patch("finance_agent.agent.Groq", return_value=mock_client):
        from finance_agent.agent import run_turn
        import os; monkeypatch.setenv("GROQ_API_KEY", "test")
        result, msgs, _ = run_turn([], "How much on groceries?")

    assert "175" in result or result
    assert any(m["role"] == "user" for m in msgs)


def test_run_turn_executes_tool_then_responds(patch_csv, monkeypatch):
    """Simulate: LLM calls tool -> tool result -> LLM gives final answer."""
    import json

    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "spending_by_category"
    tc.function.arguments = json.dumps({"month": "2026-03"})

    tool_call_resp = _mock_groq_response("", tool_calls=[tc])
    final_resp = _mock_groq_response("Groceries: $175.00")
    final_resp.choices[0].finish_reason = "stop"

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [tool_call_resp, final_resp]

    with patch("finance_agent.agent.Groq", return_value=mock_client):
        from finance_agent.agent import run_turn
        monkeypatch.setenv("GROQ_API_KEY", "test")
        result, msgs, _ = run_turn([], "How much on groceries last month?")

    assert result == "Groceries: $175.00"
    # Two LLM calls: one for tool, one for final
    assert mock_client.chat.completions.create.call_count == 2


def test_run_turn_context_prefix_injected(patch_csv, monkeypatch):
    mem_module.get_context().set(month="2026-03", category="Groceries")

    final_resp = _mock_groq_response("February groceries: $85.00")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = final_resp

    with patch("finance_agent.agent.Groq", return_value=mock_client):
        from finance_agent.agent import run_turn
        monkeypatch.setenv("GROQ_API_KEY", "test")
        ctx = mem_module.get_context()
        prefix = ctx.to_prefix()
        run_turn([], prefix + "And the month before?")

    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    user_msg = next(m for m in call_messages if m["role"] == "user")
    assert "2026-03" in user_msg["content"]


# ---------------------------------------------------------------------------
# _strip_context_update
# ---------------------------------------------------------------------------

def test_strip_context_update_extracts_fields():
    from finance_agent.main import _strip_context_update
    text = "You spent $100.\nCONTEXT_UPDATE month=2026-03 category=Dining"
    clean, month, category, account = _strip_context_update(text)
    assert "CONTEXT_UPDATE" not in clean
    assert month == "2026-03"
    assert category == "Dining"


def test_strip_context_update_no_tag():
    from finance_agent.main import _strip_context_update
    text = "You spent $100."
    clean, month, category, account = _strip_context_update(text)
    assert clean == text
    assert month is None


# ---------------------------------------------------------------------------
# Memory unit tests
# ---------------------------------------------------------------------------

def test_memory_set_and_get():
    ctx = mem_module.FinanceContext()
    ctx.set(month="2026-02", category="Dining")
    assert ctx.last_month == "2026-02"
    assert ctx.last_category == "Dining"


def test_memory_to_prefix_empty():
    assert mem_module.FinanceContext().to_prefix() == ""


def test_memory_to_prefix_with_values():
    ctx = mem_module.FinanceContext()
    ctx.set(month="2026-01", category="Groceries", account="Checking")
    p = ctx.to_prefix()
    assert "2026-01" in p and "Groceries" in p


def test_memory_clear():
    ctx = mem_module.FinanceContext()
    ctx.set(month="2026-01", category="Groceries")
    ctx.clear()
    assert ctx.last_month is None


def test_memory_prev_month():
    assert mem_module.prev_month("2026-03") == "2026-02"
    assert mem_module.prev_month("2026-01") == "2025-12"
