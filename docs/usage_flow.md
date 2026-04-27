# Usage Flow — Sample Conversation

A three-turn conversation showing how memory flows through the system.

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI (main.py)
    participant Mem as Memory (FinanceContext)
    participant Agent as LlmAgent (Gemini)
    participant Tool as Tools (tools.py)

    Note over Mem: Initial state: empty

    %% ── Turn 1 ────────────────────────────────────────────────────────────
    User->>CLI: "How much did I spend on groceries last month?"
    CLI->>Mem: to_prefix() → "" (empty)
    CLI->>Agent: Content("How much did I spend on groceries last month?")
    Agent->>Tool: spending_by_category(month="2026-03", category=None)
    Tool-->>Agent: {"Groceries": 460.24, ...}
    Agent-->>CLI: "In March 2026 you spent **$460.24** on Groceries.\nCONTEXT_UPDATE month=2026-03 category=Groceries"
    CLI->>Mem: set(month="2026-03", category="Groceries")
    CLI-->>User: "In March 2026 you spent **$460.24** on Groceries."

    Note over Mem: last_month=2026-03, last_category=Groceries

    %% ── Turn 2 ────────────────────────────────────────────────────────────
    User->>CLI: "And the month before?"
    CLI->>Mem: to_prefix() → "[Memory: last_month=2026-03, last_category=Groceries]"
    CLI->>Agent: Content("[Memory: last_month=2026-03, last_category=Groceries]\nAnd the month before?")
    Note over Agent: Resolves "month before 2026-03" → 2026-02
    Agent->>Tool: spending_by_category(month="2026-02")
    Tool-->>Agent: {"Groceries": 360.90, ...}
    Agent-->>CLI: "In February 2026 you spent **$360.90** on Groceries.\nCONTEXT_UPDATE month=2026-02 category=Groceries"
    CLI->>Mem: set(month="2026-02", category="Groceries")
    CLI-->>User: "In February 2026 you spent **$360.90** on Groceries."

    Note over Mem: last_month=2026-02, last_category=Groceries

    %% ── Turn 3 ────────────────────────────────────────────────────────────
    User->>CLI: "Am I over a $400 budget on it?"
    CLI->>Mem: to_prefix() → "[Memory: last_month=2026-02, last_category=Groceries]"
    CLI->>Agent: Content("[Memory: last_month=2026-02, last_category=Groceries]\nAm I over a $400 budget on it?")
    Note over Agent: Resolves "it" → Groceries, month → 2026-02
    Agent->>Tool: budget_check(category="Groceries", monthly_limit=400.00, month="2026-02")
    Tool-->>Agent: {over_budget: false, actual_spend: 360.90, difference: -39.10}
    Agent-->>CLI: "No — you spent **$360.90** on Groceries in February, which is **$39.10 under** your $400.00 budget."
    CLI-->>User: "No — you spent **$360.90** on Groceries in February, which is **$39.10 under** your $400.00 budget."
```

## What this demonstrates

| Turn | Memory read | Tool called | Memory written |
|------|-------------|-------------|----------------|
| 1 | — (empty) | `spending_by_category(month="2026-03")` | `last_month=2026-03, last_category=Groceries` |
| 2 | `last_month=2026-03, last_category=Groceries` | `spending_by_category(month="2026-02")` | `last_month=2026-02` (category unchanged) |
| 3 | `last_month=2026-02, last_category=Groceries` | `budget_check(category="Groceries", month="2026-02")` | unchanged |

The key insight is that the `[Memory: …]` prefix is injected **into the user
message** each turn, so Gemini sees it as part of the conversation and can
resolve anaphoric references ("it", "the month before", "same category")
without any special RAG or vector-search infrastructure.
