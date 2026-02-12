"""
FastAPI Application
REST API for GST Assistant with Cohere + MCP
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from config.config import settings
from client.llm_service import claude_service  # Using cohere but keeping variable name
from client.mcp_client import mcp_client_manager

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("="*60)
    logger.info("Starting Fintech AI Assistant with Cohere...")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Cohere Model: {settings.cohere_model}")
    logger.info("="*60)
    logger.info("Connecting to MCP server...")
    
    try:
        await mcp_client_manager.get_client()
        logger.info("✓ MCP client connected successfully")
    except Exception as e:
        logger.error(f"✗ Failed to connect to MCP server: {e}")
        logger.error("Make sure MCP server is accessible")
    
    logger.info("="*60)
    logger.info("Server ready!")
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await mcp_client_manager.close()
    logger.info("Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="Fintech AI Assistant (Cohere)",
    description="GST calculation assistant using Cohere + MCP",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's natural language query")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation messages"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Calculate GST on 10000 at 18% and show me the breakdown"
            }
        }


class ChatResponse(BaseModel):
    success: bool
    intents_detected: List[str]
    is_multi_intent: bool
    response: str
    tool_calls: List[Dict[str, Any]]
    llm_provider: str = Field(default="cohere", description="LLM provider used")
    error: Optional[str] = None


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class ToolsListResponse(BaseModel):
    tools: List[Dict[str, Any]]
    count: int


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fintech AI Assistant - GST Calculator (Powered by Cohere)",
        "version": "1.0.0",
        "llm_provider": settings.llm_provider,
        "model": settings.cohere_model,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """Health check"""
    try:
        client = await mcp_client_manager.get_client()
        return {
            "status": "healthy",
            "llm_provider": settings.llm_provider,
            "cohere_model": settings.cohere_model,
            "mcp_connected": client is not None,
            "tools_available": len(client.available_tools) if client else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "llm_provider": settings.llm_provider
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process natural language query with Cohere + MCP
    
    Handles:
    - Single intent: "Calculate GST on 10000 at 18%"
    - Multi-intent: "Calculate GST on 5000 at 12% and show breakdown"
    
    Returns natural language response with tool execution results.
    """
    try:
        logger.info(f"Chat request: {request.message[:100]}...")
        
        result = await claude_service.process_query(
            request.message,
            request.conversation_history
        )
        
        return ChatResponse(
            success=True,
            intents_detected=result["intents_detected"],
            is_multi_intent=result["is_multi_intent"],
            response=result["response"],
            tool_calls=result["tool_calls"],
            llm_provider=settings.llm_provider
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            intents_detected=[],
            is_multi_intent=False,
            response=f"I encountered an error processing your request: {str(e)}",
            tool_calls=[],
            llm_provider=settings.llm_provider,
            error=str(e)
        )


@app.get("/api/mcp/tools", response_model=ToolsListResponse)
async def list_tools():
    """
    List all available MCP tools
    
    Returns:
        List of MCP tools with their schemas
    """
    try:
        client = await mcp_client_manager.get_client()
        tools = client.available_tools
        
        return ToolsListResponse(
            tools=tools,
            count=len(tools)
        )
    
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/call-tool")
async def call_tool(request: ToolCallRequest):
    """
    Directly call an MCP tool (for testing)
    
    Args:
        request: Tool name and arguments
        
    Returns:
        Tool execution result
    """
    try:
        client = await mcp_client_manager.get_client()
        
        result = await client.call_tool(
            request.tool_name,
            request.arguments
        )
        
        if result.get("success"):
            import json
            return {
                "success": True,
                "tool": request.tool_name,
                "result": json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
            }
        else:
            return {
                "success": False,
                "tool": request.tool_name,
                "error": result.get("error")
            }
    
    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/info")
async def info():
    """
    Get API information
    
    Returns:
        API and LLM configuration info
    """
    return {
        "api_version": "1.0.0",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.cohere_model,
        "features": {
            "single_intent": True,
            "multi_intent": True,
            "natural_language": True,
            "tool_calling": True,
            "mcp_integration": True
        },
        "available_intents": [
            "calculate_gst",
            "reverse_calculate_gst",
            "gst_breakdown",
            "compare_gst_rates",
            "validate_gstin"
        ]
    }


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server...")
    logger.info(f"Using Cohere model: {settings.cohere_model}")
    
    uvicorn.run(
        "client.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )