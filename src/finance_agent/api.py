"""FastAPI web app for the Personal Finance Agent.

Run with:
    uvicorn finance_agent.api:app --reload
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
from pathlib import Path

import pandas as pd
import redis as redis_lib
import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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

# ---------------------------------------------------------------------------
# Usage / session tracking — Redis-backed with in-memory fallback
# ---------------------------------------------------------------------------
_FREE_LIMIT = 10
_sessions: dict[str, dict] = {}  # fallback when Redis is unavailable

_APP_URL = os.environ.get(
    "APP_URL", "https://finance-agent-production-c752.up.railway.app"
)

_SESSION_TTL = 60 * 60 * 24 * 90  # 90 days


def _init_redis():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        r = redis_lib.Redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


_redis = _init_redis()


def _session_key(token: str) -> str:
    return f"fa:session:{token}"


def _get_session(token: str | None) -> tuple[str, dict]:
    """Return (token, session_dict), creating a new session if needed."""
    if token:
        if _redis:
            raw = _redis.get(_session_key(token))
            if raw:
                return token, json.loads(raw)
        elif token in _sessions:
            return token, _sessions[token]

    new_token = str(uuid.uuid4())
    session = {"messages_used": 0, "is_paid": False}
    if _redis:
        _redis.setex(_session_key(new_token), _SESSION_TTL, json.dumps(session))
    else:
        _sessions[new_token] = session
    return new_token, session


def _save_session(token: str, session: dict) -> None:
    if _redis:
        _redis.setex(_session_key(token), _SESSION_TTL, json.dumps(session))
    else:
        _sessions[token] = session


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
    session_token: str | None = None


class ChatResponse(BaseModel):
    response: str
    history: list[dict]
    context: dict
    session_token: str
    messages_used: int
    limit: int
    is_paid: bool


class CheckoutRequest(BaseModel):
    session_token: str | None = None


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
    sess_token, session = _get_session(req.session_token)

    # Enforce free-tier limit.
    if session["messages_used"] >= _FREE_LIMIT and not session["is_paid"]:
        raise HTTPException(
            status_code=402,
            detail={"code": "limit_reached", "session_token": sess_token},
        )

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

    csv_token = None
    if req.file_id:
        upload_path = _UPLOADS_DIR / f"{req.file_id}.csv"
        if upload_path.exists():
            csv_token = _active_csv.set(str(upload_path))

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
        if csv_token is not None:
            _active_csv.reset(csv_token)

    ctx.set(
        month=tool_args.get("month"),
        category=tool_args.get("category"),
        account=tool_args.get("account"),
    )

    session["messages_used"] += 1
    _save_session(sess_token, session)

    return ChatResponse(
        response=_clean_response(raw),
        history=updated_history,
        context=ctx.as_dict(),
        session_token=sess_token,
        messages_used=session["messages_used"],
        limit=_FREE_LIMIT,
        is_paid=session["is_paid"],
    )


@app.get("/api/usage")
def get_usage(session_token: str | None = None) -> dict:
    if session_token:
        if _redis:
            raw = _redis.get(_session_key(session_token))
            if raw:
                s = json.loads(raw)
                return {"messages_used": s["messages_used"], "limit": _FREE_LIMIT, "is_paid": s["is_paid"]}
        elif session_token in _sessions:
            s = _sessions[session_token]
            return {"messages_used": s["messages_used"], "limit": _FREE_LIMIT, "is_paid": s["is_paid"]}
    return {"messages_used": 0, "limit": _FREE_LIMIT, "is_paid": False}


@app.post("/api/stripe/checkout")
async def create_checkout(req: CheckoutRequest) -> dict:
    secret_key = os.environ.get("STRIPE_SECRET_KEY")
    if not secret_key:
        raise HTTPException(503, "Stripe is not configured on this server.")

    stripe.api_key = secret_key
    sess_token, _ = _get_session(req.session_token)

    checkout_session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {"name": "Finance Agent — Unlimited"},
                "unit_amount": 900,
                "recurring": {"interval": "month"},
            },
            "quantity": 1,
        }],
        mode="subscription",
        client_reference_id=sess_token,
        success_url=f"{_APP_URL}?paid=1",
        cancel_url=f"{_APP_URL}?paid=0",
    )

    return {"checkout_url": checkout_session.url}


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request) -> dict:
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        raise HTTPException(503, "Webhook secret not configured.")

    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.SignatureVerificationError:
        raise HTTPException(400, "Invalid Stripe signature.")

    if event.type == "checkout.session.completed":
        ref = getattr(event.data.object, "client_reference_id", None)
        if ref:
            if _redis:
                raw = _redis.get(_session_key(ref))
                session = json.loads(raw) if raw else {"messages_used": 0, "is_paid": False}
            else:
                session = _sessions.get(ref, {"messages_used": 0, "is_paid": False})
            session["is_paid"] = True
            _save_session(ref, session)

    return {"received": True}
