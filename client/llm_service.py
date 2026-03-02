"""
Local ML Service - NO External LLM
Uses custom domain-specific ML+NLP model for intent detection
100% on-premise, no data leaves your infrastructure
"""
from typing import List, Dict, Any, Optional
import json
import logging
from ml_intent_classifier import intent_classifier
from client.mcp_client import gst_client_manager, info_client_manager, redbus_client_manager

logger = logging.getLogger(__name__)


class LocalMLService:
    """
    Local ML-based service for intent detection and tool calling.
    Replaces external LLM (Cohere) for confidential data security.
    """
    
    def __init__(self):
        self.intent_classifier = intent_classifier
        logger.info("✓ Local ML Service initialized (NO external LLM)")
    
    async def process_query(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process user query using LOCAL ML model
        
        NO data sent to external APIs
        All processing happens on-premise
        
        Args:
            user_message: User's natural language query
            conversation_history: Previous conversation (for context)
            
        Returns:
            Response with intents, tool calls, and results
        """
        logger.info(f"Processing query locally: {user_message[:100]}...")
        
        # STEP 1: ML-based intent detection (LOCAL)
        analysis = self.intent_classifier.process_query(user_message)
        
        intents_detected = analysis.get("intents_detected", [])
        tool_calls_specs = analysis.get("tool_calls", [])
        
        logger.info(f"✓ Intents detected: {intents_detected}")
        logger.info(f"✓ Tool calls: {len(tool_calls_specs)}")
        
        # STEP 2: Get MCP clients
        gst_client = await gst_client_manager.get_client()
        info_client = await info_client_manager.get_client()
        redbus_client = await redbus_client_manager.get_client()
        
        # Map tool names to clients
        tool_client_map = {}
        for tool in gst_client.available_tools:
            tool_client_map[tool["name"]] = gst_client
        for tool in redbus_client.available_tools:
            tool_client_map[tool["name"]] = redbus_client
        for tool in info_client.available_tools:
            tool_client_map[tool["name"]] = info_client
        
        # STEP 3: Execute tool calls via MCP
        mcp_results = []
        
        for idx, tool_spec in enumerate(tool_calls_specs, 1):
            tool_name = tool_spec["tool_name"]
            tool_parameters = tool_spec["parameters"]
            
            logger.info(f"[{idx}/{len(tool_calls_specs)}] Executing: {tool_name}")
            logger.info(f"  Parameters: {tool_parameters}")
            
            # Route to correct MCP client
            client = tool_client_map.get(tool_name)
            if not client:
                logger.error(f"No client for tool: {tool_name}")
                mcp_results.append({
                    "tool": tool_name,
                    "input": tool_parameters,
                    "error": f"Tool '{tool_name}' not found",
                    "success": False
                })
                continue
            
            try:
                # Execute tool via MCP
                result = await client.call_tool(tool_name, tool_parameters)
                
                if result.get("success") and result.get("result"):
                    try:
                        parsed = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    except json.JSONDecodeError:
                        parsed = result["result"]
                    
                    mcp_results.append({
                        "tool": tool_name,
                        "input": tool_parameters,
                        "result": parsed,
                        "success": True
                    })
                    
                    logger.info(f"[{idx}/{len(tool_calls_specs)}] ✓ Success")
                
                else:
                    error_msg = result.get("error", "Tool returned no result")
                    mcp_results.append({
                        "tool": tool_name,
                        "input": tool_parameters,
                        "error": error_msg,
                        "success": False
                    })
                    logger.error(f"[{idx}/{len(tool_calls_specs)}] ✗ Failed: {error_msg}")
            
            except Exception as e:
                logger.error(f"[{idx}/{len(tool_calls_specs)}] Exception: {e}")
                mcp_results.append({
                    "tool": tool_name,
                    "input": tool_parameters,
                    "error": str(e),
                    "success": False
                })
        
        # STEP 4: Generate response (template-based, NO LLM)
        response_text = self._generate_response(mcp_results, intents_detected, user_message)
        
        return {
            "success": True,
            "intents_detected": intents_detected,
            "is_multi_intent": len(intents_detected) > 1,
            "tool_calls": mcp_results,
            "response": response_text,
            "stop_reason": "complete",
            "ml_model": "local_domain_specific",
            "debug_info": {
                "total_tools_called": len(mcp_results),
                "successful_tools": len([r for r in mcp_results if r.get("success")]),
                "tools": intents_detected,
                "entities_extracted": analysis.get("entities", {})
            }
        }
    
    def _generate_response(
        self, 
        mcp_results: List[Dict], 
        intents: List[str],
        user_query: str
    ) -> str:
        """
        Generate natural language response using templates (NO LLM)
        
        Args:
            mcp_results: Results from MCP tool execution
            intents: Detected intents
            user_query: Original user query
            
        Returns:
            Natural language response string
        """
        if not mcp_results:
            return "I couldn't find any relevant information for your query. Please try rephrasing."
        
        response_parts = []
        
        for result in mcp_results:
            tool_name = result.get("tool", "")
            
            if not result.get("success"):
                error = result.get("error", "Unknown error")
                response_parts.append(f"❌ Error in {tool_name}: {error}")
                continue
            
            data = result.get("result", {})
            
            # Template-based response generation
            if tool_name == "calculate_gst":
                base = data.get("base_amount", 0)
                gst = data.get("gst_amount", 0)
                total = data.get("total_amount", 0)
                rate = data.get("gst_rate", 0)
                response_parts.append(
                    f"**GST Calculation Result:**\n"
                    f"• Base Amount: ₹{base:,.2f}\n"
                    f"• GST @ {rate}%: ₹{gst:,.2f}\n"
                    f"• Total Amount: ₹{total:,.2f}"
                )
            
            elif tool_name == "reverse_calculate_gst":
                total = data.get("total_amount", 0)
                base = data.get("base_amount", 0)
                gst = data.get("gst_amount", 0)
                rate = data.get("gst_rate", 0)
                response_parts.append(
                    f"**Reverse GST Calculation:**\n"
                    f"• Total Amount: ₹{total:,.2f}\n"
                    f"• Base Amount (excluding GST): ₹{base:,.2f}\n"
                    f"• GST Amount @ {rate}%: ₹{gst:,.2f}"
                )
            
            elif tool_name == "gst_breakdown":
                breakdown = data.get("breakdown", {})
                base = data.get("base_amount", 0)
                btype = breakdown.get("type", "Unknown")
                cgst = breakdown.get("cgst", 0)
                sgst = breakdown.get("sgst", 0)
                igst = breakdown.get("igst", 0)
                
                response_parts.append(
                    f"**GST Breakdown ({btype}):**\n"
                    f"• Base Amount: ₹{base:,.2f}\n"
                    f"• CGST: ₹{cgst:,.2f}\n"
                    f"• SGST: ₹{sgst:,.2f}\n"
                    f"• IGST: ₹{igst:,.2f}"
                )
            
            elif tool_name == "compare_gst_rates":
                base = data.get("base_amount", 0)
                comparisons = data.get("comparisons", [])
                response_parts.append(f"**GST Rate Comparison for ₹{base:,.2f}:**")
                for comp in comparisons:
                    rate = comp.get("rate", 0)
                    total = comp.get("total_amount", 0)
                    diff = comp.get("difference_from_lowest", 0)
                    response_parts.append(f"• {rate}%: ₹{total:,.2f} (+₹{diff:,.2f})")
            
            elif tool_name == "validate_gstin":
                valid = data.get("valid", False)
                gstin = data.get("gstin", "")
                components = data.get("components", {})
                
                if valid:
                    state = components.get("state_code", "")
                    pan = components.get("pan_number", "")
                    response_parts.append(
                        f"**GSTIN Validation: ✅ Valid**\n"
                        f"• GSTIN: {gstin}\n"
                        f"• State Code: {state}\n"
                        f"• PAN: {pan}"
                    )
                else:
                    error = data.get("error", "Invalid format")
                    response_parts.append(
                        f"**GSTIN Validation: ❌ Invalid**\n"
                        f"• GSTIN: {gstin}\n"
                        f"• Reason: {error}"
                    )
            
            # Onboarding info templates
            elif tool_name == "get_company_onboarding_guide":
                title = data.get("title", "Company Onboarding Guide")
                steps = data.get("steps", [])
                completion = data.get("completion_message", "")
                
                response_parts.append(f"**{title}**\n")
                for step in steps:
                    step_num = step.get("step_number", 0)
                    step_title = step.get("title", "")
                    response_parts.append(f"\n**Step {step_num}: {step_title}**")
                    
                    if "actions" in step:
                        for action in step["actions"]:
                            response_parts.append(f"  • {action}")
                    
                    if "required_fields" in step:
                        response_parts.append("  Required fields:")
                        for field in step["required_fields"][:5]:  # Show first 5
                            fname = field.get("field", "")
                            response_parts.append(f"    - {fname}")
                
                if completion:
                    response_parts.append(f"\n{completion}")
            
            elif tool_name == "get_bank_onboarding_guide":
                title = data.get("title", "Bank Onboarding Guide")
                steps = data.get("steps", [])
                completion = data.get("completion_message", "")
                
                response_parts.append(f"**{title}**\n")
                for step in steps:
                    step_num = step.get("step_number", 0)
                    step_title = step.get("title", "")
                    response_parts.append(f"\n**Step {step_num}: {step_title}**")
                    
                    if "actions" in step:
                        for action in step["actions"]:
                            response_parts.append(f"  • {action}")
                
                if completion:
                    response_parts.append(f"\n{completion}")
            
            elif tool_name == "get_vendor_onboarding_guide":
                title = data.get("title", "Vendor Onboarding Guide")
                steps = data.get("steps", [])
                completion = data.get("completion_message", "")
                
                response_parts.append(f"**{title}**\n")
                for step in steps:
                    step_num = step.get("step_number", 0)
                    step_title = step.get("title", "")
                    response_parts.append(f"\n**Step {step_num}: {step_title}**")
                
                if completion:
                    response_parts.append(f"\n{completion}")
            
            elif tool_name == "get_supported_banks":
                total = data.get("total_banks", 0)
                banks = data.get("banks", [])
                response_parts.append(f"**Supported Banks ({total} total):**")
                for bank in banks[:10]:  # Show first 10
                    name = bank.get("name", "")
                    ifsc = bank.get("ifsc_prefix", "")
                    response_parts.append(f"• {name} (IFSC: {ifsc}****)")
            
            elif tool_name == "get_validation_formats":
                formats = data.get("formats", {})
                response_parts.append("**Validation Formats:**")
                for doc_type, doc_data in list(formats.items())[:5]:  # Show first 5
                    pattern = doc_data.get("pattern", "")
                    example = doc_data.get("example", "")
                    response_parts.append(f"• {doc_type}: {pattern}")
                    if example:
                        response_parts.append(f"  Example: {example}")
            
            elif tool_name in ["get_onboarding_faq", "get_common_errors", "get_company_required_documents"]:
                # Generic template for structured data
                title = data.get("title", "Information")
                response_parts.append(f"**{title}**")
                response_parts.append("(Detailed information available - please check the full response)")
            elif tool_name in ["redbus_search_redirect", "redbus_booking_redirect", "redbus_offers_redirect", "redbus_tracking_redirect", "get_popular_routes","open_redbus"]:
                # RedBus-specific templates
                if tool_name == "redbus_search_redirect":
                    response_parts.append("I've found the best bus options for your route and date. Click here to view them: [RedBus Search Results](https://www.redbus.in/)")
                elif tool_name == "redbus_booking_redirect":
                    response_parts.append("You can complete your bus booking here: [RedBus Booking Page](https://www.redbus.in/)")
                elif tool_name == "redbus_offers_redirect":
                    response_parts.append("Check out the latest offers on RedBus: [RedBus Offers](https://www.redbus.in/offers)")
                elif tool_name == "redbus_tracking_redirect":
                    response_parts.append("Track your bus in real-time here: [RedBus Tracking](https://www.redbus.in/track-bus)")
                elif tool_name == "get_popular_routes":
                    routes = data.get("popular_routes", [])
                    response_parts.append("Here are some popular bus routes:")
                    for route in routes[:10]:  # Show first 10
                        source = route.get("source", "")
                        destination = route.get("destination", "")
                        response_parts.append(f"• {source} → {destination}")
                elif tool_name == "open_redbus":
                    base = "https://your-api-domain.com"   # your FastAPI server
                    redirect_to = result["input"].get("redirect_to", "web")
                    url = f"{base}/redbus"
                    if redirect_to == "app":
                        url += "?platform=app"
                    response_parts.append(f"**Opening RedBus 🚌**\n[Click here to open RedBus]({url})")
            else:
                # Fallback for unknown tools
                response_parts.append(f"✓ {tool_name} executed successfully")
        
        return "\n\n".join(response_parts)


# Global service instance
claude_service = LocalMLService()  # Keep same variable name for compatibility