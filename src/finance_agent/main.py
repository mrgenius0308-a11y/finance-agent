"""Interactive CLI for the Personal Finance Agent (Groq backend).

Run with:
    python -m finance_agent.main
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv(Path(__file__).parents[2] / ".env")

from finance_agent.agent import run_turn, _REGISTRY
from finance_agent.data import load_csv
from finance_agent.memory import get_context, update_from_tool_args
import re

console = Console()

_CTX_RE = re.compile(
    r"CONTEXT_UPDATE\s*(?:month=(\S+))?\s*(?:category=(\S+))?\s*(?:account=(.+))?",
    re.IGNORECASE,
)


def _strip_context_update(text: str) -> tuple[str, str | None, str | None, str | None]:
    month = category = account = None
    clean = []
    for line in text.splitlines():
        m = _CTX_RE.search(line)
        if m:
            month = m.group(1) or month
            category = m.group(2) or category
            account = m.group(3) or account
        else:
            clean.append(line)
    return "\n".join(clean).strip(), month, category, account


def _print_banner() -> None:
    df = load_csv()
    t = Table.grid(padding=(0, 1))
    t.add_row("[bold cyan]Rows:[/]", str(len(df)))
    t.add_row("[bold cyan]Date range:[/]",
              f"{df['date'].min().date()} -> {df['date'].max().date()}")
    t.add_row("[bold cyan]Accounts:[/]", ", ".join(sorted(df["account"].unique())))
    from finance_agent.agent import MODEL
    t.add_row("[bold cyan]Model:[/]", f"{MODEL} via Groq")
    console.print(Panel(t, title="[bold green]Personal Finance Agent[/]",
                        subtitle="Type your question or 'exit' to quit",
                        border_style="green"))


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]ERROR:[/] GROQ_API_KEY not set. Add it to .env")
        sys.exit(1)

    _print_banner()
    messages: list[dict] = []

    while True:
        try:
            user_input = console.input("\n[bold yellow]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            console.print("[dim]Goodbye![/]")
            break

        ctx = get_context()
        prefix = ctx.to_prefix()
        full_input = prefix + user_input if prefix else user_input

        with console.status("[dim]Thinking...[/]", spinner="dots"):
            try:
                raw, messages, tool_args = run_turn(messages, full_input)
            except Exception as exc:
                console.print(f"[bold red]Error:[/] {exc}")
                continue

        # Update memory from tool args (month/category the model resolved)
        ctx.set(
            month=tool_args.get("month") or None,
            category=tool_args.get("category") or None,
        )

        console.print("\n[bold green]Agent:[/]")
        console.print(Markdown(raw))


if __name__ == "__main__":
    main()
