"""Finance agent with tool calling — works with any OpenAI-compatible backend.

Supports Groq (cloud) and Ollama (local) via the same openai client interface.
Each call to run_turn() executes a full ReAct loop: LLM -> tool -> LLM -> ...
until the model produces a final text response.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from finance_agent.tools import (
    budget_check,
    load_transactions,
    monthly_summary,
    recurring_subscriptions,
    search_transactions,
    spending_by_category,
    top_merchants,
)

# Defaults (used by CLI and as fallback)
DEFAULT_MODEL    = "llama-3.3-70b-versatile"
GROQ_BASE_URL    = "https://api.groq.com/openai/v1"
OLLAMA_BASE_URL  = "http://localhost:11434/v1"

# Model menus surfaced to the UI
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

OLLAMA_MODELS = [
    "llama3.2",
    "llama3.1:8b",
    "qwen2.5:7b",
    "mistral",
    "phi3",
    "deepseek-r1:7b",
]

import datetime as _dt
_TODAY = _dt.date.today().strftime("%Y-%m-%d")
_LAST_FULL_MONTH = (
    _dt.date.today().replace(day=1) - _dt.timedelta(days=1)
).strftime("%Y-%m")

SYSTEM_PROMPT = f"""You are a careful personal finance assistant. Today is {_TODAY}.

Transaction data covers 2025-11 through 2026-04.
The most recent COMPLETE calendar month is {_LAST_FULL_MONTH}. Always use this when the user says "last month" and no month is specified.

Spending categories in the data: Groceries, Dining, Transport, Utilities, Rent, Entertainment, Subscriptions, Shopping, Health, Income.
Bank accounts in the data: Checking, Credit Card.

Rules:
- ALWAYS call a tool before answering. Never invent numbers.
- The CSV is already loaded — NEVER ask the user for a file path. Just call the tool.
- For questions about expense types/categories, call spending_by_category (default month) and list the category names returned.
- NEVER pass the `account` parameter unless the user explicitly says "in my Checking account" or "on my Credit Card". Omit it by default.
- The `account` parameter means bank account type ("Checking" or "Credit Card"), NOT a spending category.
- Format money as USD with 2 decimals ($1,234.56).
- Resolve "that month"/"month before"/"same category" from the [Memory: ...] prefix.
- If a tool returns empty, report it plainly and suggest what info you need."""

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Any] = {
    "load_transactions": load_transactions,
    "spending_by_category": spending_by_category,
    "top_merchants": top_merchants,
    "monthly_summary": monthly_summary,
    "recurring_subscriptions": recurring_subscriptions,
    "budget_check": budget_check,
    "search_transactions": search_transactions,
}

# ---------------------------------------------------------------------------
# JSON schemas for function calling
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "load_transactions",
            "description": "Load transactions from CSV and return a summary (row count, date range, accounts).",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Optional CSV path override."}
            }, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spending_by_category",
            "description": "Return total spending per category for a given month. Amounts are positive USD.",
            "parameters": {"type": "object", "properties": {
                "month": {"type": "string", "description": "Month as YYYY-MM string e.g. \"2026-03\". Defaults to last full month."},
                "category": {"type": "string", "description": "Optional spending category filter e.g. 'Groceries', 'Dining'."},
            }, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_merchants",
            "description": "Return top merchants by spend for a month. Pass month='all' for full history.",
            "parameters": {"type": "object", "properties": {
                "month": {"type": "string", "description": "YYYY-MM or 'all'."},
                "n": {"type": "integer", "description": "Number of merchants to return (default 5)."},
            }, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "monthly_summary",
            "description": "Return total income, spend, net, and top 3 categories for a month.",
            "parameters": {"type": "object", "properties": {
                "month": {"type": "string", "description": "YYYY-MM. Defaults to last full month."},
            }, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recurring_subscriptions",
            "description": "Detect recurring subscriptions: same merchant, similar amount, 3+ months.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "budget_check",
            "description": "Check if spending in a category exceeds a monthly budget limit.",
            "parameters": {"type": "object", "properties": {
                "category": {"type": "string", "description": "Category name e.g. 'Dining'."},
                "monthly_limit": {"type": "number", "description": "Budget limit in USD."},
                "month": {"type": "string", "description": "YYYY-MM. Defaults to last full month."},
            }, "required": ["category", "monthly_limit"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transactions",
            "description": "Case-insensitive substring search on description/merchant.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search keyword."},
                "month": {"type": "string", "description": "Optional YYYY-MM filter."},
            }, "required": ["query"]},
        },
    },
]


def _execute_tool(name: str, args: dict) -> str:
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        if "month" in args and args["month"] is not None:
            args["month"] = str(args["month"])
        result = fn(**args)
        return json.dumps(result, default=str)
    except Exception as exc:
        return f"Tool error: {exc}"


def run_turn(
    messages: list[dict],
    user_input: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = GROQ_BASE_URL,
    api_key: str | None = None,
) -> tuple[str, list[dict], dict]:
    """Run one conversation turn and return (response_text, updated_messages, last_tool_args).

    Works with any OpenAI-compatible backend (Groq, Ollama, etc.) via base_url.
    """
    if api_key is None:
        api_key = os.environ["GROQ_API_KEY"]

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Strip tool-call internals from prior turns — Groq rejects them when
    # passed back as history. The final assistant text already encodes the
    # tool results in natural language, so context is preserved.
    clean_history = [
        m for m in messages
        if m.get("role") in ("user", "assistant") and "tool_calls" not in m
    ]
    # Keep only the last 10 messages (~5 turns) to avoid 413 "request too large"
    # errors on models with low TPM limits (e.g. Groq free tier).
    clean_history = clean_history[-10:]
    messages = clean_history + [{"role": "user", "content": user_input}]
    last_tool_args: dict = {}

    while True:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    max_tokens=2048,
                )
                break  # success
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                err = str(exc)
                if status in (429, 413) or "rate_limit" in err or "too large" in err:
                    # Back off and retry on rate-limit / oversized-request errors
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                    if attempt == 2:
                        raise
                elif "tool_use_failed" in err or status == 400:
                    # Groq rejected the model's malformed tool call — retry without tools
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                        max_tokens=2048,
                    )
                    break
                else:
                    raise

        choice = response.choices[0]
        messages = messages + [{"role": "assistant", "content": choice.message.content or ""}]

        if choice.finish_reason == "tool_calls" and (choice.message.tool_calls or []):
            messages[-1] = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ],
            }
            tool_results = []
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments or "{}") or {}
                last_tool_args.update(args)
                result = _execute_tool(tc.function.name, args)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            messages = messages + tool_results
        else:
            final_text = choice.message.content or ""
            clean_messages = [
                m for m in messages
                if m.get("role") in ("user", "assistant") and "tool_calls" not in m
            ]
            return final_text, clean_messages, last_tool_args
