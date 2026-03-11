"""
client/main.py — Bank AI Assistant  v3.1.0
Production-ready FastAPI application.

Changes from v3.0.0:
  - Wired to agent_manager (adds memory + PostgreSQL persistence)
  - ChatRequest: session_id + user_id fields added
  - Request timeout (30 s) on /api/chat
  - CORS restricted to ALLOWED_ORIGINS env var
  - latency_ms calculated from real processing_time
  - /health extended: agent readiness + DB health
  - New endpoints: /api/session/{id}/context|history, DELETE /api/session/{id}
  - New endpoint:  /api/analytics/intents
"""
from dotenv import load_dotenv
load_dotenv()   # must be first — manager.py reads env vars at initialize() time

import asyncio
import importlib.util
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from config.config import settings
from client.mcp_client import bank_client_manager, gst_client_manager, info_client_manager
from manager import agent_manager          # ← NEW: replaces direct claude_service usage

logging.basicConfig(
    level  = getattr(logging, settings.log_level.upper()),
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Optional query logger ──────────────────────────────────────────────
_qlog_spec = importlib.util.find_spec("query_logger")
if _qlog_spec is not None:
    from query_logger import metrics_router, query_logger
    _query_logging = True
else:
    metrics_router = None
    query_logger   = None
    _query_logging = False

# ── Config ─────────────────────────────────────────────────────────────
REQUEST_TIMEOUT_SECS = int(os.getenv("REQUEST_TIMEOUT_SECS", "30"))

# CORS: restrict in production — set ALLOWED_ORIGINS="https://app.yourdomain.com"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: List[str] = (
    ["*"] if _raw_origins == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)


# ══════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Bank AI Assistant v3.1.0")
    logger.info(f"LLM Provider : {settings.llm_provider}")
    logger.info("=" * 60)

    # ── MCP clients ───────────────────────────────────────────────
    for name, mgr in (
        ("Bank MCP Server ", bank_client_manager),
        ("GST Calculator  ", gst_client_manager),
        ("Onboarding Info ", info_client_manager),
    ):
        try:
            client = await mgr.get_client()
            logger.info(f"✓ {name}: {len(client.available_tools)} tools")
        except Exception as e:
            logger.error(f"✗ {name}: {e}")

    # ── Agent (memory + PostgreSQL) ───────────────────────────────
    await agent_manager.initialize()

    if _query_logging:
        logger.info("✓ Query logging enabled → logs/queries.jsonl")

    logger.info("=" * 60)
    logger.info("Bank AI Assistant ready!")
    logger.info("=" * 60)

    yield

    # ── Graceful shutdown ─────────────────────────────────────────
    logger.info("Shutting down...")
    await agent_manager.shutdown()
    await bank_client_manager.close()
    await gst_client_manager.close()
    await info_client_manager.close()
    logger.info("Goodbye!")


# ══════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "Bank AI Assistant",
    description = (
        "AI-powered banking assistant — Payments, B2B, GST, EPF, ESIC, "
        "Payroll, Taxes, Insurance, Custom/SEZ, Bank Statement, "
        "Account Management, Transactions, Dues & Support."
    ),
    version     = "3.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins      = ALLOWED_ORIGINS,   # ← restricted; set ALLOWED_ORIGINS in .env
    allow_credentials  = True,
    allow_methods      = ["*"],
    allow_headers      = ["*"],
)

if metrics_router:
    app.include_router(metrics_router)
    logger.info("✓ /metrics endpoints registered")


# ══════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [
            {"message": "Show my account balance",
             "session_id": "sess-abc-123", "user_id": "user-001"},
            {"message": "Pay EPF and ESIC dues for 02-2026",
             "session_id": "sess-abc-123", "user_id": "user-001"},
        ]}
    )
    message:    str           = Field(...,            description="User's natural language banking query")
    session_id: Optional[str] = Field(default=None,  description="Session ID — auto-generated if omitted")
    user_id:    Optional[str] = Field(default=None,  description="Authenticated user ID — 'anonymous' if omitted")
    # kept for backwards compatibility (ignored when agent_manager is active)
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None)


class ChatResponse(BaseModel):
    success:          bool
    intents_detected: List[str]
    is_multi_intent:  bool
    response:         str
    tool_calls:       List[Dict[str, Any]]
    llm_provider:     str            = "local_ml"
    session_id:       Optional[str] = None
    context_used:     Optional[bool] = None
    memory_snapshot:  Optional[Dict] = None
    error:            Optional[str]  = None


# ══════════════════════════════════════════════════════════════════════
# STANDARD ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "message":      "Bank AI Assistant",
        "version":      "3.1.0",
        "llm_provider": settings.llm_provider,
        "docs":         "/docs",
        "health":       "/health",
    }


@app.get("/health")
async def health():
    """
    Full health check:
      - MCP server connectivity
      - Agent readiness
      - PostgreSQL connectivity
    """
    server_status: Dict[str, Any] = {}
    total_tools = 0

    for name, mgr in (
        ("bank", bank_client_manager),
        ("gst",  gst_client_manager),
        ("info", info_client_manager),
    ):
        try:
            client = await mgr.get_client()
            server_status[name] = {"connected": True, "tools": len(client.available_tools)}
            total_tools += len(client.available_tools)
        except Exception as e:
            server_status[name] = {"connected": False, "error": str(e)}

    return {
        "status":        "healthy",
        "version":       "3.1.0",
        "llm_provider":  settings.llm_provider,
        "total_tools":   total_tools,
        "agent_ready":   agent_manager.is_ready(),          # ← NEW
        "db_healthy":    await agent_manager.db_health(),   # ← NEW
        "query_logging": _query_logging,
        "cors_origins":  ALLOWED_ORIGINS,
        "servers":       server_status,
        "storage":       await agent_manager.storage_stats(),  # ← NEW
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN CHAT ENDPOINT
# ══════════════════════════════════════════════════════════════════════
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — handles all banking queries with session memory.

    Pass the same session_id across multiple requests to enable context
    carry-over (e.g. company_id / gstin remembered between turns).
    """
    session_id = (request.session_id or str(uuid.uuid4())).strip()
    user_id    = (request.user_id    or "anonymous").strip()

    logger.info(f"[Chat] session={session_id} user={user_id} msg={request.message[:80]!r}")

    try:
        # ── Timeout guard — prevents MCP slow calls hanging the server ──
        result = await asyncio.wait_for(
            agent_manager.process(
                message    = request.message,
                session_id = session_id,
                user_id    = user_id,
            ),
            timeout = REQUEST_TIMEOUT_SECS,
        )

        processing_ms = int(result.get("processing_time", 0) * 1000)

        if _query_logging:
            query_logger.log_query(
                query      = request.message,
                intents    = result["intents_detected"],
                tools      = [t["tool"] for t in result.get("tool_calls", [])],
                latency_ms = processing_ms,    # ← real value now
                success    = True,
            )

        return ChatResponse(
            success          = True,
            intents_detected = result["intents_detected"],
            is_multi_intent  = result["is_multi_intent"],
            response         = result["response"],
            tool_calls       = result["tool_calls"],
            llm_provider     = settings.llm_provider,
            session_id       = session_id,
            context_used     = result.get("context_used"),
            memory_snapshot  = result.get("memory_snapshot"),
        )

    except asyncio.TimeoutError:
        logger.error(f"[Chat] Timeout after {REQUEST_TIMEOUT_SECS}s — session={session_id}")
        if _query_logging:
            query_logger.log_query(
                query=request.message, intents=[], tools=[],
                latency_ms=REQUEST_TIMEOUT_SECS * 1000,
                success=False, error="timeout",
            )
        return ChatResponse(
            success=False, intents_detected=[], is_multi_intent=False,
            response="Request timed out. Please try again.",
            tool_calls=[], llm_provider=settings.llm_provider,
            session_id=session_id, error="timeout",
        )

    except Exception as e:
        logger.error(f"[Chat] Error session={session_id}: {e}", exc_info=True)
        if _query_logging:
            query_logger.log_query(
                query=request.message, intents=[], tools=[],
                latency_ms=0, success=False, error=str(e),
            )
        return ChatResponse(
            success=False, intents_detected=[], is_multi_intent=False,
            response="I encountered an error. Please try again.",
            tool_calls=[], llm_provider=settings.llm_provider,
            session_id=session_id, error=str(e),
        )


# ══════════════════════════════════════════════════════════════════════
# SESSION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/session/{session_id}/context")
async def get_session_context(session_id: str):
    """
    Return what the agent currently knows about a session.
    Useful for debugging memory / context injection.
    """
    if not agent_manager.is_ready():
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {
        "session_id": session_id,
        "context":    agent_manager.get_context(session_id),
    }


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Return in-memory conversation history for a session."""
    if not agent_manager.is_ready():
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {
        "session_id": session_id,
        "history":    agent_manager.get_history(session_id),
    }


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session memory — call on user logout."""
    if not agent_manager.is_ready():
        raise HTTPException(status_code=503, detail="Agent not ready")
    agent_manager.clear_session(session_id)
    return {"session_id": session_id, "cleared": True}


# ══════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/analytics/intents")
async def intent_analytics(days: int = 7):
    """Intent frequency + avg confidence for the past N days (from PostgreSQL)."""
    return {
        "days":  days,
        "stats": await agent_manager.intent_stats(days=days),
    }


# ══════════════════════════════════════════════════════════════════════
# API INFO
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/info")
async def info():
    return {
        "api_version":  "3.1.0",
        "llm_provider": settings.llm_provider,
        "platform":     "Bank AI Assistant",
        "features": {
            "core_payments":        True,
            "bulk_upload_payment":  True,
            "b2b":                  True,
            "insurance":            True,
            "bank_statement":       True,
            "custom_sez":           True,
            "gst":                  True,
            "esic":                 True,
            "epf":                  True,
            "payroll":              True,
            "taxes":                True,
            "account_management":   True,
            "transaction_history":  True,
            "dues_reminders":       True,
            "dashboard_analytics":  True,
            "company_management":   True,
            "support":              True,
            "multi_intent":         True,
            "natural_language":     True,
            "mcp_integration":      True,
            "session_memory":       True,
            "query_logging":        _query_logging,
        },
        "server": {
            "name":   "Bank AI Assistant",
            "module": "mcp_server.data_server",
            "tools": [
                # Core Payment
                "initiate_payment", "get_payment_status", "cancel_payment",
                "retry_payment", "get_payment_receipt", "validate_beneficiary",
                # Upload Payment
                "upload_bulk_payment", "validate_payment_file",
                # B2B
                "onboard_business_partner", "send_invoice", "get_received_invoices",
                "acknowledge_payment", "create_proforma_invoice",
                "create_cd_note", "create_purchase_order",
                # Insurance
                "fetch_insurance_dues", "pay_insurance_premium", "get_insurance_payment_history",
                # Bank Statement
                "fetch_bank_statement", "download_bank_statement",
                "get_account_balance", "get_transaction_history",
                # Custom / SEZ
                "pay_custom_duty", "track_custom_duty_payment", "get_custom_duty_history",
                # GST
                "fetch_gst_dues", "pay_gst", "create_gst_challan", "get_gst_payment_history",
                # ESIC
                "fetch_esic_dues", "pay_esic", "get_esic_payment_history",
                # EPF
                "fetch_epf_dues", "pay_epf", "get_epf_payment_history",
                # Payroll
                "fetch_payroll_summary", "process_payroll", "get_payroll_history",
                # Taxes
                "fetch_tax_dues", "pay_direct_tax", "pay_state_tax",
                "pay_bulk_tax", "get_tax_payment_history",
                # Account Management
                "get_account_summary", "get_account_details",
                "get_linked_accounts", "set_default_account",
                # Transaction & History
                "search_transactions", "get_transaction_details",
                "download_transaction_report", "get_pending_transactions",
                # Dues & Reminders
                "get_upcoming_dues", "get_overdue_payments", "set_payment_reminder",
                "get_reminder_list", "delete_reminder",
                # Dashboard & Analytics
                "get_dashboard_summary", "get_spending_analytics",
                "get_cashflow_summary", "get_monthly_report", "get_vendor_payment_summary",
                # Company Management
                "get_company_profile", "update_company_details", "get_gst_profile",
                "get_authorized_signatories", "manage_user_roles",
                # Support
                "raise_support_ticket", "get_ticket_history",
                "chat_with_support", "get_contact_details",
            ]
        }
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL LISTING
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/mcp/tools")
async def list_all_tools():
    """List all available tools from all MCP servers."""
    all_tools: Dict[str, Any] = {}
    total = 0
    for name, mgr in (
        ("bank", bank_client_manager),
        ("gst",  gst_client_manager),
        ("info", info_client_manager),
    ):
        try:
            client = await mgr.get_client()
            all_tools[name] = client.available_tools
            total += len(client.available_tools)
        except Exception as e:
            all_tools[name] = {"error": str(e)}
    return {"total_tools": total, "servers": all_tools}


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "client.main:app",
        host   = settings.host,
        port   = settings.port,
        reload = settings.debug,
    )