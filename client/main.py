
"""
FastAPI Application — Bank AI Assistant
Payments, B2B, GST, EPF, ESIC, Payroll, Taxes,
Insurance, Custom/SEZ, Bank Statement,
Account Management, Transactions, Dues, Dashboard & Support.

Uses Local ML (no external LLM) + single Bank MCP Server (data_server.py).
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import logging
import importlib.util
from contextlib import asynccontextmanager

from config.config import settings
from client.llm_service import claude_service
from client.mcp_client import bank_client_manager, gst_client_manager, info_client_manager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Optional query logger
_qlog_spec = importlib.util.find_spec("query_logger")
if _qlog_spec is not None:
    from query_logger import metrics_router, query_logger
    _query_logging = True
else:
    metrics_router = None
    query_logger   = None
    _query_logging = False


# ═══════════════════════════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Bank AI Assistant")
    logger.info(f"LLM Provider : {settings.llm_provider}")
    logger.info("=" * 60)

    try:
        bank_client = await bank_client_manager.get_client()
        logger.info(f"✓ Bank MCP Server  : {len(bank_client.available_tools)} tools")
    except Exception as e:
        logger.error(f"✗ Bank MCP Server failed: {e}")

    try:
        gst_client = await gst_client_manager.get_client()
        logger.info(f"✓ GST Calculator   : {len(gst_client.available_tools)} tools")
    except Exception as e:
        logger.error(f"✗ GST Calculator failed: {e}")

    try:
        info_client = await info_client_manager.get_client()
        logger.info(f"✓ Onboarding Info  : {len(info_client.available_tools)} tools")
    except Exception as e:
        logger.error(f"✗ Onboarding Info failed: {e}")

    if _query_logging:
        logger.info("✓ Query logging enabled → logs/queries.jsonl")
    else:
        logger.info("ℹ  Query logging disabled")

    logger.info("=" * 60)
    logger.info("Bank AI Assistant ready!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")
    await bank_client_manager.close()
    await gst_client_manager.close()
    await info_client_manager.close()
    logger.info("Goodbye!")


# ═══════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Bank AI Assistant",
    description=(
        "AI-powered banking assistant covering Payments, B2B, GST, EPF, ESIC, "
        "Payroll, Taxes, Insurance, Custom/SEZ, Bank Statement, "
        "Account Management, Transactions, Dues & Support."
    ),
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if metrics_router:
    app.include_router(metrics_router)
    logger.info("✓ /metrics endpoints registered")


# ═══════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"message": "Initiate NEFT payment of 50000 to BENE001"},
                {"message": "What is my account balance?"},
                {"message": "Pay GST for GSTIN 27ABCDE1234F1Z5"},
                {"message": "Process payroll for February 2026"},
                {"message": "Show upcoming dues for next 30 days"},
                {"message": "Upload bulk payment file"},
                {"message": "Send invoice to partner PART001 for 1,00,000"},
                {"message": "Pay EPF and ESIC dues for 02-2026"},
                {"message": "Get dashboard summary and spending analytics"},
            ]
        }
    )
    message: str = Field(..., description="User's natural language banking query")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None)


class ChatResponse(BaseModel):
    success: bool
    intents_detected: List[str]
    is_multi_intent: bool
    response: str
    tool_calls: List[Dict[str, Any]]
    llm_provider: str = Field(default="local_ml")
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# STANDARD ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "message":      "Bank AI Assistant",
        "version":      "3.0.0",
        "llm_provider": settings.llm_provider,
        "model":        "local_ml",
        "features": [
            "Core Payments", "Bulk Upload Payments",
            "B2B (Invoice, PO, CD Note, Proforma)",
            "Insurance", "Bank Statement",
            "Custom Duty / SEZ",
            "GST (Fetch, Pay, Challan)",
            "ESIC", "EPF", "Payroll",
            "Taxes (Direct, State, Bulk)",
            "Account Management",
            "Transaction History & Search",
            "Dues & Reminders",
            "Dashboard & Analytics",
            "Company Management",
            "Support & Communication",
        ],
        "docs":   "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    server_status = {}
    total_tools   = 0

    try:
        bank_client = await bank_client_manager.get_client()
        server_status["bank"] = {
            "connected": True,
            "tools": len(bank_client.available_tools),
        }
        total_tools += len(bank_client.available_tools)
    except Exception as e:
        server_status["bank"] = {"connected": False, "error": str(e)}

    try:
        gst_client = await gst_client_manager.get_client()
        server_status["gst"] = {
            "connected": True,
            "tools": len(gst_client.available_tools),
        }
        total_tools += len(gst_client.available_tools)
    except Exception as e:
        server_status["gst"] = {"connected": False, "error": str(e)}

    try:
        info_client = await info_client_manager.get_client()
        server_status["info"] = {
            "connected": True,
            "tools": len(info_client.available_tools),
        }
        total_tools += len(info_client.available_tools)
    except Exception as e:
        server_status["info"] = {"connected": False, "error": str(e)}

    return {
        "status":        "healthy",
        "llm_provider":  settings.llm_provider,
        "total_tools":   total_tools,
        "query_logging": _query_logging,
        "servers":       server_status,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — handles all banking queries.

    Supported query categories:
    - Payments     : "Initiate NEFT payment of 50000 to vendor"
    - Bulk Payment : "Upload bulk payment file"
    - B2B          : "Send invoice to partner for 1 lakh"
    - Insurance    : "Show insurance premium dues"
    - Bank Stmt    : "Get statement for Jan 2026"
    - Custom/SEZ   : "Pay custom duty for BOE12345"
    - GST          : "Pay GST for GSTIN 27ABCDE1234F1Z5"
    - ESIC         : "Pay ESIC dues for 02-2026"
    - EPF          : "Fetch EPF dues for establishment PF001"
    - Payroll      : "Process payroll for February 2026"
    - Taxes        : "Pay TDS for PAN ABCDE1234F"
    - Account      : "Show account balance and linked accounts"
    - Transactions : "Search transactions for last month"
    - Dues         : "What dues are coming up in next 30 days?"
    - Dashboard    : "Show dashboard summary and cashflow"
    - Company      : "Show company profile and GST numbers"
    - Support      : "Raise a support ticket for payment issue"
    - Multi-intent : "Pay EPF and ESIC dues for 02-2026 and show dashboard"
    """
    try:
        logger.info(f"Chat: {request.message[:100]}...")

        result = await claude_service.process_query(
            request.message,
            request.conversation_history
        )

        if _query_logging:
            query_logger.log_query(
                query      = request.message,
                intents    = result["intents_detected"],
                tools      = [t["tool"] for t in result.get("tool_calls", [])],
                latency_ms = 0,
                success    = True,
            )

        return ChatResponse(
            success          = True,
            intents_detected = result["intents_detected"],
            is_multi_intent  = result["is_multi_intent"],
            response         = result["response"],
            tool_calls       = result["tool_calls"],
            llm_provider     = settings.llm_provider,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)

        if _query_logging:
            query_logger.log_query(
                query      = request.message,
                intents    = [],
                tools      = [],
                latency_ms = 0,
                success    = False,
                error      = str(e),
            )

        return ChatResponse(
            success          = False,
            intents_detected = [],
            is_multi_intent  = False,
            response         = f"Error: {str(e)}",
            tool_calls       = [],
            llm_provider     = settings.llm_provider,
            error            = str(e),
        )


@app.get("/api/mcp/tools")
async def list_all_tools():
    """List all available tools from all MCP servers."""
    all_tools = {"bank": [], "gst": [], "info": []}
    total     = 0

    try:
        bank_client       = await bank_client_manager.get_client()
        all_tools["bank"] = bank_client.available_tools
        total            += len(bank_client.available_tools)
    except Exception as e:
        all_tools["bank"] = {"error": str(e)}

    try:
        gst_client       = await gst_client_manager.get_client()
        all_tools["gst"] = gst_client.available_tools
        total           += len(gst_client.available_tools)
    except Exception as e:
        all_tools["gst"] = {"error": str(e)}

    try:
        info_client       = await info_client_manager.get_client()
        all_tools["info"] = info_client.available_tools
        total            += len(info_client.available_tools)
    except Exception as e:
        all_tools["info"] = {"error": str(e)}

    return {
        "total_tools": total,
        "servers":     all_tools,
    }


@app.get("/api/info")
async def info():
    return {
        "api_version":  "3.0.0",
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


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "client.main:app",
        host   = settings.host,
        port   = settings.port,
        reload = settings.debug
    )