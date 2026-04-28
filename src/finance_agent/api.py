"""FastAPI web app for the Personal Finance Agent.

Run with:
    uvicorn finance_agent.api:app --reload
"""

from __future__ import annotations

import io
import os
import re
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).parents[2] / ".env")

from finance_agent.agent import (
    GROQ_BASE_URL,
    GROQ_MODELS,
    OLLAMA_BASE_URL,
    OLLAMA_MODELS,
    DEFAULT_MODEL,
    run_turn,
)
from finance_agent.data import _active_csv
from finance_agent.memory import FinanceContext

_STATIC_DIR  = Path(__file__).parents[2] / "static"
_UPLOADS_DIR = Path(__file__).parents[2] / "uploads"
_UPLOADS_DIR.mkdir(exist_ok=True)

_REQUIRED_COLS = {"date", "description", "merchant", "category", "amount", "account"}

app = FastAPI(title="Finance Agent")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

_CTX_RE = re.compile(
    r"CONTEXT_UPDATE\s*(?:month=(\S+))?\s*(?:category=(\S+))?\s*(?:account=(.+))?",
    re.IGNORECASE,
)

# Matches leaked function-call markup that small models sometimes emit in text,
# e.g. "<function=top_merchants>{"n": 5}</function>" or without the opening "<".
_FUNC_TAG_RE = re.compile(r"<?\bfunction=[^>\n]+>[^<\n]*</function>", re.IGNORECASE)


def _strip_context_update(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _CTX_RE.search(line)
    ).strip()


def _clean_response(text: str) -> str:
    text = _FUNC_TAG_RE.sub("", text)
    return _strip_context_update(text)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    type: str = "groq"                    # "groq" | "ollama"
    model: str = DEFAULT_MODEL
    ollama_url: str = OLLAMA_BASE_URL     # only used when type == "ollama"


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    context: dict = {}
    file_id: str | None = None
    provider: ProviderConfig = ProviderConfig()


class ChatResponse(BaseModel):
    response: str
    history: list[dict]
    context: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/providers")
def get_providers() -> dict:
    """Return available providers and their model lists for the UI."""
    groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "groq": {
            "label": "Groq (cloud)",
            "base_url": GROQ_BASE_URL,
            "models": GROQ_MODELS,
            "default_model": GROQ_MODELS[0],
            "available": groq_key_set,
            "unavailable_reason": None if groq_key_set else "GROQ_API_KEY not set",
        },
        "ollama": {
            "label": "Ollama (local)",
            "base_url": OLLAMA_BASE_URL,
            "models": OLLAMA_MODELS,
            "default_model": OLLAMA_MODELS[0],
            "available": True,          # we can't know without trying
            "unavailable_reason": None,
        },
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV (.csv) and Excel (.xlsx / .xls) files are supported.")

    content = await file.read()

    try:
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise HTTPException(400, f"Could not parse file: {exc}")

    df.columns = df.columns.str.strip().str.lower()

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise HTTPException(
            400,
            f"Missing required columns: {', '.join(sorted(missing))}. "
            f"Expected: {', '.join(sorted(_REQUIRED_COLS))}.",
        )

    try:
        df["date"]   = pd.to_datetime(df["date"])
        df["amount"] = pd.to_numeric(df["amount"])
    except Exception as exc:
        raise HTTPException(400, f"Could not parse date/amount columns: {exc}")

    file_id   = str(uuid.uuid4())
    save_path = _UPLOADS_DIR / f"{file_id}.csv"
    df.to_csv(save_path, index=False)

    return {
        "file_id":  file_id,
        "filename": file.filename,
        "rows":     len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "accounts": sorted(df["account"].astype(str).unique().tolist()),
    }


@app.get("/api/download/{file_id}", response_model=None)
def download_file(file_id: str, format: str = "csv") -> StreamingResponse | FileResponse:
    if not all(c in "0123456789abcdef-" for c in file_id):
        raise HTTPException(400, "Invalid file ID.")

    path = _UPLOADS_DIR / f"{file_id}.csv"
    if not path.exists():
        raise HTTPException(404, "File not found.")

    if format == "xlsx":
        df  = pd.read_csv(path)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=transactions.xlsx"},
        )

    return FileResponse(
        path,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transactions.csv"},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Resolve provider credentials server-side — never trust the client with keys.
    if req.provider.type == "ollama":
        base_url = req.provider.ollama_url
        api_key  = "ollama"          # Ollama ignores the key but openai client requires one
    else:
        base_url = GROQ_BASE_URL
        api_key  = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise HTTPException(500, "GROQ_API_KEY is not set on the server.")

    ctx = FinanceContext(
        last_month=req.context.get("last_month"),
        last_category=req.context.get("last_category"),
        last_account=req.context.get("last_account"),
    )

    prefix     = ctx.to_prefix()
    full_input = (prefix + req.message) if prefix else req.message

    token = None
    if req.file_id:
        upload_path = _UPLOADS_DIR / f"{req.file_id}.csv"
        if upload_path.exists():
            token = _active_csv.set(str(upload_path))

    try:
        raw, updated_history, tool_args = run_turn(
            req.history,
            full_input,
            model=req.provider.model,
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if token is not None:
            _active_csv.reset(token)

    ctx.set(
        month=tool_args.get("month"),
        category=tool_args.get("category"),
        account=tool_args.get("account"),
    )

    return ChatResponse(
        response=_clean_response(raw),
        history=updated_history,
        context=ctx.as_dict(),
    )
