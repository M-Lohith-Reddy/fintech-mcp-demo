"""
FastAPI Application - FINAL VERSION
GST Calculator + Onboarding Info with Local ML + 2 MCP Servers
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse                        # ADD: Option C redirects
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import logging
import urllib.parse                                                    # ADD: URL encoding
from contextlib import asynccontextmanager
from datetime import datetime, date as dt_date                        # ADD: date formatting

from config.config import settings
from client.llm_service import claude_service
from client.mcp_client import gst_client_manager, info_client_manager, redbus_client_manager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional query logger
import importlib.util
_qlog_spec = importlib.util.find_spec("query_logger")
if _qlog_spec is not None:
    from query_logger import metrics_router, query_logger
    _query_logging = True
else:
    metrics_router  = None
    query_logger    = None
    _query_logging  = False


# ═══════════════════════════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Fintech AI Assistant - FINAL")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info("=" * 60)

    try:
        gst_client = await gst_client_manager.get_client()
        logger.info(f"✓ GST Server: {len(gst_client.available_tools)} tools")
    except Exception as e:
        logger.error(f"✗ GST Server failed: {e}")

    try:
        info_client = await info_client_manager.get_client()
        logger.info(f"✓ Info Server: {len(info_client.available_tools)} tools")
    except Exception as e:
        logger.error(f"✗ Info Server failed: {e}")

    try:
        redbus_client = await redbus_client_manager.get_client()
        logger.info(f"✓ RedBus Server: {len(redbus_client.available_tools)} tools")
    except Exception as e:
        logger.error(f"✗ RedBus Server failed: {e}")

    if _query_logging:
        logger.info("✓ Query logging enabled → logs/queries.jsonl")
    else:
        logger.info("ℹ  Query logging disabled (add query_logger.py to enable)")

    logger.info("=" * 60)
    logger.info("All servers ready!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")
    await gst_client_manager.close()
    await info_client_manager.close()
    await redbus_client_manager.close()
    logger.info("Goodbye!")


# ═══════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Fintech AI Assistant",
    description="GST Calculator + Onboarding Information + RedBus Redirect with Local ML + MCP",
    version="2.0.0",
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
                {"message": "Calculate GST on 10000 at 18%"},
                {"message": "How to register my company on Vanghee B2B?"},
                {"message": "What are the bank onboarding steps?"},
                {"message": "Book a bus from Bangalore to Mumbai tomorrow"},
                {"message": "Show me RedBus offers from Bangalore"}
            ]
        }
    )
    message: str = Field(..., description="User's natural language query")
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
        "message":      "Fintech AI Assistant",
        "version":      "2.0.0",
        "llm_provider": settings.llm_provider,
        "model":        "local_ml",
        "features":     ["GST Calculations", "Onboarding Information", "RedBus Redirect"],
        "platform":     "Vanghee B2B",
        "docs":         "/docs",
        "health":       "/health"
    }


@app.get("/health")
async def health():
    server_status = {}
    total_tools   = 0

    try:
        gst_client = await gst_client_manager.get_client()
        server_status["gst"] = {"connected": True, "tools": len(gst_client.available_tools)}
        total_tools += len(gst_client.available_tools)
    except Exception as e:
        server_status["gst"] = {"connected": False, "error": str(e)}

    try:
        info_client = await info_client_manager.get_client()
        server_status["info"] = {"connected": True, "tools": len(info_client.available_tools)}
        total_tools += len(info_client.available_tools)
    except Exception as e:
        server_status["info"] = {"connected": False, "error": str(e)}

    try:
        redbus_client = await redbus_client_manager.get_client()
        server_status["redbus"] = {"connected": True, "tools": len(redbus_client.available_tools)}
        total_tools += len(redbus_client.available_tools)
    except Exception as e:
        server_status["redbus"] = {"connected": False, "error": str(e)}

    return {
        "status":        "healthy",
        "llm_provider":  settings.llm_provider,
        "total_tools":   total_tools,
        "query_logging": _query_logging,
        "servers":       server_status
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — handles GST calculations, onboarding information and RedBus redirects.

    Supported queries:
    - GST       : "Calculate GST on 10000 at 18%"
    - Onboarding: "How to register my company?"
    - Bank      : "Bank onboarding steps"
    - Vendor    : "How to register vendor?"
    - RedBus    : "Book bus from Bangalore to Mumbai tomorrow"
    - RedBus    : "Show RedBus offers from Chennai"
    - RedBus    : "Track my bus TIN123456789"
    - Multi     : "Calculate GST and book bus from Bangalore to Pune"
    - Multi     : "Calculate GST and show company onboarding guide"
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
            llm_provider     = settings.llm_provider
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
            error            = str(e)
        )


@app.get("/api/mcp/tools")
async def list_all_tools():
    """List all available tools from all MCP servers"""
    all_tools = {"gst": [], "info": [], "redbus": []}
    total     = 0

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

    try:
        redbus_client       = await redbus_client_manager.get_client()
        all_tools["redbus"] = redbus_client.available_tools
        total              += len(redbus_client.available_tools)
    except Exception as e:
        all_tools["redbus"] = {"error": str(e)}

    return {"total_tools": total, "servers": all_tools}


@app.get("/api/info")
async def info():
    return {
        "api_version": "2.0.0",
        "llm_provider": settings.llm_provider,
        "platform":    "Vanghee B2B",
        "features": {
            "gst_calculations":       True,
            "onboarding_information": True,
            "redbus_redirect":        True,
            "multi_intent":           True,
            "natural_language":       True,
            "mcp_integration":        True,
            "query_logging":          _query_logging,
            "gstin_api":              importlib.util.find_spec("gstin_validator") is not None,
        },
        "servers": {
            "gst": {
                "tools": [
                    "calculate_gst", "reverse_calculate_gst",
                    "gst_breakdown", "compare_gst_rates", "validate_gstin"
                ]
            },
            "info": {
                "tools": [
                    "get_company_onboarding_guide", "get_company_required_documents",
                    "get_bank_onboarding_guide",    "get_supported_banks",
                    "get_vendor_onboarding_guide",  "get_validation_formats",
                    "get_onboarding_faq",           "get_common_errors"
                ]
            },
            "redbus": {
                "tools": [
                    "redbus_search_redirect", "redbus_booking_redirect",
                    "redbus_offers_redirect", "redbus_tracking_redirect",
                    "get_popular_routes",     "open_redbus"
                ]
            }
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# OPTION C — DIRECT BROWSER REDIRECT ENDPOINTS
# ───────────────────────────────────────────────────────────────────────
# These are plain GET endpoints that perform an HTTP 302 redirect
# straight to the correct RedBus URL.  No frontend JS needed — just
# point an <a href> or window.location at one of these URLs.
#
# Endpoint map:
#   GET /redbus                              → redbus.in homepage
#   GET /redbus?platform=app                → redbus://home  (deep link)
#   GET /redbus/search?src=X&dst=Y          → search results
#   GET /redbus/search?src=X&dst=Y&date=D   → search results on date
#   GET /redbus/offers                      → offers page
#   GET /redbus/offers?city=Bangalore       → city-filtered offers
#   GET /redbus/booking/{tin}               → booking details
#   GET /redbus/track/{tin}                 → live tracking
# ═══════════════════════════════════════════════════════════════════════

@app.get(
    "/redbus",
    summary="Open RedBus homepage",
    tags=["RedBus Redirects"],
    response_class=RedirectResponse
)
async def redbus_home(platform: Optional[str] = None):
    """
    Redirect to RedBus homepage (web or app deep link).

    - /redbus              → https://www.redbus.in
    - /redbus?platform=app → redbus://home
    """
    url = "redbus://home" if platform == "app" else "https://www.redbus.in"
    logger.info(f"[Redirect] /redbus → {url}")
    return RedirectResponse(url=url, status_code=302)


@app.get(
    "/redbus/search",
    summary="Search buses on RedBus",
    tags=["RedBus Redirects"],
    response_class=RedirectResponse
)
async def redbus_search(
    src:  str,
    dst:  str,
    date: Optional[str] = None   # YYYY-MM-DD
):
    """
    Redirect to RedBus search results for a given route and date.

    - /redbus/search?src=Bangalore&dst=Mumbai
    - /redbus/search?src=Bangalore&dst=Mumbai&date=2026-03-15
    """
    if date:
        try:
            redbus_date = datetime.strptime(date, "%Y-%m-%d").strftime("%d-%b-%Y")
        except ValueError:
            redbus_date = dt_date.today().strftime("%d-%b-%Y")
    else:
        redbus_date = dt_date.today().strftime("%d-%b-%Y")

    slug = (
        f"{src.strip().lower().replace(' ', '-')}"
        f"-to-"
        f"{dst.strip().lower().replace(' ', '-')}"
    )
    url = f"https://www.redbus.in/bus-tickets/{slug}?doj={redbus_date}"
    logger.info(f"[Redirect] /redbus/search → {url}")
    return RedirectResponse(url=url, status_code=302)


@app.get(
    "/redbus/offers",
    summary="View RedBus offers",
    tags=["RedBus Redirects"],
    response_class=RedirectResponse
)
async def redbus_offers(city: Optional[str] = None):
    """
    Redirect to RedBus offers page, optionally filtered by departure city.

    - /redbus/offers
    - /redbus/offers?city=Bangalore
    """
    url = "https://www.redbus.in/offers"
    if city:
        url += f"?src={urllib.parse.quote(city.strip())}"
    logger.info(f"[Redirect] /redbus/offers → {url}")
    return RedirectResponse(url=url, status_code=302)


@app.get(
    "/redbus/booking/{tin}",
    summary="View RedBus booking by TIN",
    tags=["RedBus Redirects"],
    response_class=RedirectResponse
)
async def redbus_booking(tin: str):
    """
    Redirect to RedBus booking details for a given TIN number.

    - /redbus/booking/TIN123456789
    """
    url = f"https://www.redbus.in/mybookings/ticket-details?tin={tin.strip().upper()}"
    logger.info(f"[Redirect] /redbus/booking/{tin} → {url}")
    return RedirectResponse(url=url, status_code=302)


@app.get(
    "/redbus/track/{tin}",
    summary="Track a live bus journey by TIN",
    tags=["RedBus Redirects"],
    response_class=RedirectResponse
)
async def redbus_track(tin: str):
    """
    Redirect to RedBus live bus tracking for a given TIN number.

    - /redbus/track/TIN123456789
    """
    url = f"https://www.redbus.in/mybookings/track-my-bus?tin={tin.strip().upper()}"
    logger.info(f"[Redirect] /redbus/track/{tin} → {url}")
    return RedirectResponse(url=url, status_code=302)


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