"""
20-question robustness test for the finance agent.

Runs directly against run_turn() — no server needed.
Tests: tool-calling, multi-turn follow-ups, casual messages, edge cases.

Usage:
    cd finance_agent
    .venv/Scripts/activate
    python -m pytest tests/test_robustness.py -v -s
  OR run standalone:
    python tests/test_robustness.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Make sure the src package is importable when run standalone
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[1] / ".env")

from finance_agent.agent import run_turn, GROQ_BASE_URL, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Test session definition
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    label: str
    message: str
    # If True, continues the previous conversation (same history).
    # If False, starts a fresh session.
    continues: bool = True
    # Optional substring that MUST appear in the response (case-insensitive).
    must_contain: str = ""


SESSIONS: list[list[Turn]] = [

    # ── Session A: multi-turn finance conversation (10 turns) ───────────────
    [
        Turn("01 highest spend",
             "What is the highest spending category last month?",
             continues=False,
             must_contain="$"),

        Turn("02 follow-up account",
             "Which account type made those transactions?",
             must_contain=""),   # can't predict exact text

        Turn("03 top merchants",
             "Show me my top 5 merchants by spend this month.",
             must_contain="$"),

        Turn("04 groceries",
             "How much did I spend on groceries last month?",
             must_contain="$"),

        Turn("05 compare dining",
             "How does that compare to my dining expenses?",
             must_contain=""),

        Turn("06 subscriptions",
             "Do I have any recurring subscriptions?",
             must_contain=""),

        Turn("07 biggest subscription",
             "Which one is the most expensive?",
             must_contain=""),

        Turn("08 budget check",
             "Did I go over a $400 budget for dining last month?",
             must_contain=""),

        Turn("09 casual thanks",
             "Thanks, that's really helpful!",
             must_contain=""),   # casual — no tool needed

        Turn("10 goodbye",
             "Bye!",
             must_contain=""),
    ],

    # ── Session B: specific months + search (5 turns) ───────────────────────
    [
        Turn("11 march summary",
             "Give me a summary of March 2026.",
             continues=False,
             must_contain="2026"),       # model may say "March 2026" not "2026-03"

        Turn("12 november summary",
             "Now show me November 2025.",
             must_contain="2025"),       # model may say "November 2025" not "2025-11"

        Turn("13 search starbucks",
             "Search for any Starbucks transactions.",
             must_contain=""),

        Turn("14 search rent",
             "Find all rent payments in the data.",
             must_contain=""),

        Turn("15 income question",
             "What was my total income last month?",
             must_contain="$"),
    ],

    # ── Session C: edge cases + adversarial inputs (5 turns) ────────────────
    [
        Turn("16 vague question",
             "How am I doing?",
             continues=False,
             must_contain=""),

        Turn("17 out-of-range month",
             "What did I spend in January 2020?",  # outside the data range
             must_contain=""),

        Turn("18 nonsense input",
             "asdfghjkl",
             must_contain=""),

        Turn("19 multi-question",
             "What are my top merchants AND my total spend AND my subscriptions all in one answer?",
             must_contain=""),

        Turn("20 net worth question",
             "Based on my transactions, what is my overall financial health?",
             must_contain=""),
    ],
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class Result:
    turn: Turn
    response: str = ""
    error: str = ""
    duration_ms: int = 0

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.turn.must_contain and self.turn.must_contain.lower() not in self.response.lower():
            return False
        return True

    @property
    def status(self) -> str:
        if self.error:
            return "FAIL [X]"
        if self.turn.must_contain and self.turn.must_contain.lower() not in self.response.lower():
            return "WARN [!]"
        return "PASS [.]"


def run_session(session: list[Turn], model: str, base_url: str, api_key: str) -> list[Result]:
    results: list[Result] = []
    history: list[dict] = []

    for turn in session:
        if not turn.continues:
            history = []

        start = time.monotonic()
        try:
            response_text, history, _ = run_turn(
                history,
                turn.message,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            ms = int((time.monotonic() - start) * 1000)
            results.append(Result(turn=turn, response=response_text, duration_ms=ms))
        except Exception:
            ms = int((time.monotonic() - start) * 1000)
            results.append(Result(turn=turn, error=traceback.format_exc(), duration_ms=ms))
            history = []  # reset on error so subsequent turns can continue

    return results


def run_all() -> list[Result]:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to .env or set it as an env variable.")
        sys.exit(1)

    all_results: list[Result] = []
    for session in SESSIONS:
        all_results.extend(run_session(session, DEFAULT_MODEL, GROQ_BASE_URL, api_key))

    return all_results


def print_report(results: list[Result]) -> None:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)

    print("\n" + "=" * 70)
    print(f"  ROBUSTNESS REPORT — {passed}/{total} passed")
    print("=" * 70)

    for r in results:
        label = f"[{r.turn.label}]"
        q_preview = r.turn.message[:55] + ("…" if len(r.turn.message) > 55 else "")
        print(f"\n{r.status}  {label:25}  {r.duration_ms:>5}ms")
        print(f"   Q: {q_preview}")
        if r.error:
            # Show just the last line of the traceback
            last_line = [l for l in r.error.strip().splitlines() if l.strip()][-1]
            print(f"   ERROR: {last_line}")
        else:
            a_preview = r.response[:120].replace("\n", " ")
            if len(r.response) > 120:
                a_preview += "…"
            print(f"   A: {a_preview}")

    print("\n" + "=" * 70)
    print(f"  {passed}/{total} passed  |  {total - passed} failed/warned")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# pytest-compatible test so it works with `pytest -v -s` too
# ---------------------------------------------------------------------------

def test_robustness():
    results = run_all()
    print_report(results)
    failures = [r for r in results if not r.passed]
    assert not failures, f"{len(failures)} turn(s) failed:\n" + "\n".join(
        f"  [{r.turn.label}] {r.error or 'must_contain not found'}" for r in failures
    )


if __name__ == "__main__":
    results = run_all()
    print_report(results)
    sys.exit(0 if all(r.passed for r in results) else 1)
