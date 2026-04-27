# Personal Finance Agent

An interactive CLI that answers natural-language questions about your bank
transactions, powered by **Google ADK** and **Gemini 1.5 Flash**.

```
You: How much did I spend on groceries last month?
Agent: In March 2026 you spent $460.24 on Groceries.

You: And the month before?
Agent: In February 2026 you spent $360.90 on Groceries.

You: Am I over a $400 budget on it?
Agent: No — you spent $360.90 on Groceries in February, $39.10 under budget.
```

## Architecture & diagrams

- [Architecture overview](docs/architecture.md) — component flowchart
- [Usage flow](docs/usage_flow.md) — 3-turn sequence diagram with memory

## Setup

### 1. Clone and create the virtual environment

```bash
cd finance_agent
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and paste your Google AI Studio key:
# GOOGLE_API_KEY=AIza...
```

Get a free key at <https://aistudio.google.com/app/apikey>.

### 4. Run the agent

```bash
python -m finance_agent.main
```

### 5. Run tests

```bash
pytest
```

## Using your own CSV

Set `FINANCE_CSV_PATH` in `.env` to the path of your CSV file:

```
FINANCE_CSV_PATH=/path/to/my_transactions.csv
```

The CSV must have these columns (header row required):

| Column | Format | Notes |
|--------|--------|-------|
| `date` | `YYYY-MM-DD` | |
| `description` | string | Free-text description |
| `merchant` | string | Merchant/payee name |
| `category` | string | Spending category |
| `amount` | float | Negative = spend, positive = income |
| `account` | string | e.g. `Checking`, `Credit Card` |

## Available questions (examples)

- *"How much did I spend on dining last month?"*
- *"And the month before?"*  ← memory follow-up
- *"What are my top 5 merchants in March?"*
- *"List my recurring subscriptions."*
- *"Am I over a $500 budget on groceries?"*
- *"Search for any Uber charges in January."*
- *"Give me a summary of February."*

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GOOGLE_API_KEY is not set` | Add your key to `.env` |
| `ModuleNotFoundError: finance_agent` | Run `pip install -e .` with the venv active |
| `FileNotFoundError` for CSV | Check `FINANCE_CSV_PATH` in `.env` |
| Gemini quota errors | The free tier has RPM limits; wait a moment and retry |

## API note

This project targets **google-adk ≥ 1.0** (stable).  The ADK 2.0 beta
introduced a `NodeRunner` / `BaseNode` architecture; if you upgrade to 2.0,
the `Runner` import path and `LlmAgent` constructor may change.  See the
[ADK changelog](https://github.com/google/adk-python/releases) for details.
