"""Conversation context memory for the finance agent.

Stores the last month, category, and account the user was discussing so
the agent can resolve follow-up questions like "and the month before?" or
"same category".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FinanceContext:
    last_month: str | None = None
    last_category: str | None = None
    last_account: str | None = None

    def set(
        self,
        month: str | None = None,
        category: str | None = None,
        account: str | None = None,
    ) -> None:
        if month:
            self.last_month = month
        if category:
            self.last_category = category
        if account:
            self.last_account = account

    def clear(self) -> None:
        self.last_month = None
        self.last_category = None
        self.last_account = None

    def to_prefix(self) -> str:
        """Format context as a bracketed prefix injected before the user message."""
        parts: list[str] = []
        if self.last_month:
            parts.append(f"last_month={self.last_month}")
        if self.last_category:
            parts.append(f"last_category={self.last_category}")
        if self.last_account:
            parts.append(f"last_account={self.last_account}")
        if not parts:
            return ""
        return "[Memory: " + ", ".join(parts) + "]\n"

    def as_dict(self) -> dict:
        return {
            "last_month": self.last_month,
            "last_category": self.last_category,
            "last_account": self.last_account,
        }


# Module-level singleton — shared across the CLI session
_ctx = FinanceContext()


def get_context() -> FinanceContext:
    return _ctx


def reset_context() -> None:
    global _ctx
    _ctx = FinanceContext()


def update_from_tool_args(func_name: str, args: dict) -> None:
    """Update context from arguments the LLM passed to a tool call."""
    month = args.get("month")
    category = args.get("category")
    account = args.get("account")
    _ctx.set(month=month, category=category, account=account)


def prev_month(month: str) -> str:
    """Return the month before *month* (format 'YYYY-MM')."""
    import pandas as pd
    p = pd.Period(month, "M") - 1
    return str(p)
