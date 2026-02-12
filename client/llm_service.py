"""
Cohere LLM Service with MCP Integration
Handles intent detection, single and multi-intent queries using Cohere
FIXED: Proper multi-intent handling
"""
import cohere
from typing import List, Dict, Any, Optional
import json
import logging
from config.config import settings
from client.mcp_client import mcp_client_manager

logger = logging.getLogger(__name__)


class CohereLLMService:
    """Service for Cohere LLM with MCP tool integration"""
    
    def __init__(self):
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        self.model = settings.cohere_model
    
    async def process_query(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process user query with Cohere + MCP tools
        
        Handles:
        - Single intent queries
        - Multi-intent queries
        - Parameter extraction
        - Tool execution via MCP
        
        Args:
            user_message: User's natural language query
            conversation_history: Previous conversation messages
            
        Returns:
            Response with intents, tool calls, and natural language response
        """
        logger.info(f"Processing query with Cohere: {user_message[:100]}...")
        
        # Get MCP client and tools
        mcp_client = await mcp_client_manager.get_client()
        tools = mcp_client.get_tools_for_cohere()
        
        logger.info(f"Available MCP tools: {len(tools)}")
        
        # Build chat history for Cohere
        chat_history = []
        if conversation_history:
            for msg in conversation_history:
                role = "USER" if msg["role"] == "user" else "CHATBOT"
                chat_history.append({
                    "role": role,
                    "message": msg["content"]
                })
        
        # IMPROVED System preamble with explicit multi-intent instructions
        preamble = """You are a helpful GST (Goods and Services Tax) assistant for India.

You have access to tools for GST calculations. Use them to help users with:
- Calculating GST amounts and totals
- Reverse calculating base amounts from totals
- Getting detailed GST breakdowns (CGST, SGST, IGST)
- Comparing different GST rates
- Validating GSTIN numbers

CRITICAL MULTI-INTENT INSTRUCTIONS:
When users ask for MULTIPLE operations in a SINGLE query, you MUST call MULTIPLE tools.

Examples of multi-intent queries:
1. "Calculate GST on 5000 at 12% and show breakdown"
   → Call BOTH: calculate_gst AND gst_breakdown

2. "Calculate GST on 10000 at 18% and also compare with 12%"
   → Call BOTH: calculate_gst AND compare_gst_rates

3. "Show me the breakdown and also validate GSTIN 29ABCDE1234F1Z5"
   → Call BOTH: gst_breakdown AND validate_gstin

4. "Calculate GST, show breakdown, and compare rates"
   → Call ALL THREE tools

IMPORTANT RULES:
✓ If user says "and", "also", "then", "additionally" → USE MULTIPLE TOOLS
✓ Each distinct request needs its own tool call
✓ Don't summarize - actually call each tool separately
✓ Call tools in the order requested
✓ Use the exact parameter values from the query

When responding:
1. Identify ALL intents in the user's message
2. Call the appropriate tool for EACH intent
3. Format currency in Indian Rupees (₹)
4. Provide clear, organized responses
5. Show results for each operation separately

Always be accurate and call all necessary tools."""

        try:
            # Call Cohere API with tools
            response = self.client.chat(
                message=user_message,
                model=self.model,
                preamble=preamble,
                chat_history=chat_history,
                tools=tools,
                temperature=0.1,  # Lower temperature for more deterministic tool calling
                force_single_step=False  # Allow multiple tool calls
            )
            
            logger.info(f"Cohere response received")
            logger.info(f"Tool calls in response: {len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0}")
            
            # Process response
            return await self._process_response(response, user_message, mcp_client, tools, preamble)
        
        except Exception as e:
            logger.error(f"Error processing query with Cohere: {e}")
            raise
    
    async def _process_response(
        self,
        response: Any,
        user_message: str,
        mcp_client: Any,
        tools: List[Dict[str, Any]],
        preamble: str
    ) -> Dict[str, Any]:
        """Process Cohere's response and handle tool calls"""
        
        tool_calls = []
        text_response = response.text if response.text else ""
        intents_detected = []
        mcp_results = []
        
        # Check if Cohere wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            num_tools = len(response.tool_calls)
            logger.info(f"✓ Cohere requested {num_tools} tool call(s)")
            logger.info(f"✓ Multi-intent detected: {num_tools > 1}")
            
            for idx, tool_call in enumerate(response.tool_calls, 1):
                tool_name = tool_call.name
                tool_parameters = tool_call.parameters
                
                intents_detected.append(tool_name)
                
                logger.info(f"[{idx}/{num_tools}] Executing MCP tool: {tool_name}")
                logger.info(f"[{idx}/{num_tools}] Parameters: {tool_parameters}")
                
                # Execute MCP tool
                result = await mcp_client.call_tool(
                    tool_name,
                    tool_parameters
                )
                
                # Parse result (MCP returns JSON string)
                if result.get("success") and result.get("result"):
                    try:
                        parsed_result = json.loads(result["result"])
                    except json.JSONDecodeError:
                        parsed_result = result["result"]
                    
                    mcp_results.append({
                        "tool": tool_name,
                        "input": tool_parameters,
                        "result": parsed_result,
                        "success": True
                    })
                    
                    tool_calls.append({
                        "name": tool_name,
                        "parameters": tool_parameters,
                        "result": parsed_result
                    })
                    
                    logger.info(f"[{idx}/{num_tools}] ✓ Success: {tool_name}")
                else:
                    error_msg = result.get("error", "Unknown error")
                    mcp_results.append({
                        "tool": tool_name,
                        "input": tool_parameters,
                        "error": error_msg,
                        "success": False
                    })
                    
                    logger.error(f"[{idx}/{num_tools}] ✗ Failed: {tool_name} - {error_msg}")
            
            # CRITICAL: Get final response from Cohere with ALL tool results
            if tool_calls:
                logger.info(f"Sending {len(tool_calls)} tool result(s) back to Cohere...")
                
                # Prepare tool results for Cohere
                tool_results = []
                for i, tool_call in enumerate(response.tool_calls):
                    if mcp_results[i]["success"]:
                        tool_results.append({
                            "call": tool_call,
                            "outputs": [mcp_results[i]["result"]]
                        })
                    else:
                        tool_results.append({
                            "call": tool_call,
                            "outputs": [{"error": mcp_results[i]["error"]}]
                        })
                
                # Get final response with tool results
                try:
                    final_response = self.client.chat(
                        message="",  # Empty message for tool result continuation
                        model=self.model,
                        preamble=preamble,  # Include preamble for context
                        chat_history=[
                            {"role": "USER", "message": user_message},
                            {"role": "CHATBOT", "message": text_response}
                        ],
                        tools=tools,
                        tool_results=tool_results,
                        temperature=0.3
                    )
                    
                    text_response = final_response.text
                    logger.info("✓ Got final response from Cohere with tool results")
                    
                except Exception as e:
                    logger.error(f"Error getting final response: {e}")
                    # Fallback: Create response manually
                    text_response = self._create_fallback_response(mcp_results, user_message)
        
        else:
            # No tools were called
            logger.warning(f"⚠ No tools called by Cohere for query: {user_message[:100]}")
            logger.info("Using direct Cohere response without tool execution")
        
        return {
            "success": True,
            "intents_detected": intents_detected,
            "is_multi_intent": len(intents_detected) > 1,
            "tool_calls": mcp_results,
            "response": text_response,
            "stop_reason": "complete",
            "debug_info": {
                "total_tools_called": len(intents_detected),
                "tools": intents_detected
            }
        }
    
    def _create_fallback_response(self, mcp_results: List[Dict], user_message: str) -> str:
        """Create a fallback response if Cohere fails to generate one"""
        if not mcp_results:
            return "I processed your request but encountered an issue generating a response."
        
        response_parts = []
        
        for idx, result in enumerate(mcp_results, 1):
            tool_name = result.get("tool", "unknown")
            
            if result.get("success"):
                tool_result = result.get("result", {})
                
                # Format based on tool type
                if tool_name == "calculate_gst":
                    base = tool_result.get("base_amount", 0)
                    gst = tool_result.get("gst_amount", 0)
                    total = tool_result.get("total_amount", 0)
                    rate = tool_result.get("gst_rate", 0)
                    response_parts.append(
                        f"**GST Calculation:**\n"
                        f"- Base Amount: ₹{base:,.2f}\n"
                        f"- GST ({rate}%): ₹{gst:,.2f}\n"
                        f"- Total Amount: ₹{total:,.2f}"
                    )
                
                elif tool_name == "gst_breakdown":
                    breakdown = tool_result.get("breakdown", {})
                    response_parts.append(
                        f"**GST Breakdown ({breakdown.get('type', 'Unknown')}):**\n"
                        f"- CGST: ₹{breakdown.get('cgst', 0):,.2f}\n"
                        f"- SGST: ₹{breakdown.get('sgst', 0):,.2f}\n"
                        f"- IGST: ₹{breakdown.get('igst', 0):,.2f}"
                    )
                
                elif tool_name == "reverse_calculate_gst":
                    base = tool_result.get("base_amount", 0)
                    gst = tool_result.get("gst_amount", 0)
                    total = tool_result.get("total_amount", 0)
                    response_parts.append(
                        f"**Reverse Calculation:**\n"
                        f"- Total Amount: ₹{total:,.2f}\n"
                        f"- Base Amount: ₹{base:,.2f}\n"
                        f"- GST Amount: ₹{gst:,.2f}"
                    )
                
                elif tool_name == "compare_gst_rates":
                    comparisons = tool_result.get("comparisons", [])
                    response_parts.append("**Rate Comparison:**")
                    for comp in comparisons:
                        response_parts.append(
                            f"- {comp.get('rate')}%: Total ₹{comp.get('total_amount', 0):,.2f}"
                        )
                
                elif tool_name == "validate_gstin":
                    valid = tool_result.get("valid", False)
                    gstin = tool_result.get("gstin", "")
                    response_parts.append(
                        f"**GSTIN Validation:**\n"
                        f"- GSTIN: {gstin}\n"
                        f"- Status: {'✓ Valid' if valid else '✗ Invalid'}"
                    )
            
            else:
                response_parts.append(f"**Error in {tool_name}:** {result.get('error', 'Unknown error')}")
        
        return "\n\n".join(response_parts)


# Global service instance
claude_service = CohereLLMService()  # Keep same name for compatibility