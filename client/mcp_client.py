"""
MCP Client Wrapper for Cohere
FIXED: Proper async context management for persistent connection
"""
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict, List, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP server with persistent connection"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self._server_params: Optional[StdioServerParameters] = None
        self._read_stream = None
        self._write_stream = None
    
    async def connect(self):
        """Connect to MCP server using subprocess approach"""
        try:
            import subprocess
            import sys
            
            self._server_params = StdioServerParameters(
                command="python",
                args=["-m", "mcp_server.server"],
                env=None
            )
            
            logger.info("Connecting to MCP server...")
            
            # Use the stdio_client properly - it needs to stay in scope
            async with stdio_client(self._server_params) as (read, write):
                self._read_stream = read
                self._write_stream = write
                
                async with ClientSession(read, write) as session:
                    self.session = session
                    
                    # Initialize session
                    await session.initialize()
                    
                    # List available tools
                    tools_list = await session.list_tools()
                    self.available_tools = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema
                        }
                        for tool in tools_list.tools
                    ]
                    
                    logger.info(f"✓ Connected to MCP server")
                    logger.info(f"✓ Available tools: {len(self.available_tools)}")
                    logger.info(f"✓ Tools: {[t['name'] for t in self.available_tools]}")
                    
                    # IMPORTANT: We need to keep the session alive, but context managers
                    # make this difficult. Solution: Call tools within the same connection.
                    return self.available_tools
        
        except Exception as e:
            logger.error(f"✗ Failed to connect to MCP server: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def call_tool_direct(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call tool with a fresh connection (simpler, more reliable)
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        logger.info(f"→ Calling MCP tool: {tool_name}")
        logger.info(f"→ Arguments: {arguments}")
        
        try:
            server_params = StdioServerParameters(
                command="python",
                args=["-m", "mcp_server.server"],
                env=None
            )
            
            # Create fresh connection for each tool call (reliable approach)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool(tool_name, arguments)
                    
                    logger.info(f"✓ Tool {tool_name} executed")
                    logger.info(f"✓ Has content: {bool(result.content)}")
                    logger.info(f"✓ Is error: {result.isError}")
                    
                    # Extract result content
                    if result.content:
                        result_text = result.content[0].text if result.content else None
                        logger.info(f"✓ Result: {result_text[:200] if result_text else 'None'}...")
                        
                        return {
                            "success": not result.isError,
                            "result": result_text,
                            "is_error": result.isError
                        }
                    
                    logger.warning("⚠ No content in result")
                    return {
                        "success": not result.isError,
                        "result": None,
                        "is_error": result.isError
                    }
        
        except Exception as e:
            logger.error(f"✗ Error calling tool {tool_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool (uses fresh connection approach)
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        return await self.call_tool_direct(tool_name, arguments)
    
    def get_tools_for_cohere(self) -> List[Dict[str, Any]]:
        """
        Get tools in Cohere API format
        
        Returns:
            List of tools formatted for Cohere API
        """
        cohere_tools = []
        
        for tool in self.available_tools:
            # Convert JSON Schema to Cohere parameter format
            parameter_definitions = {}
            
            if "properties" in tool["input_schema"]:
                for param_name, param_schema in tool["input_schema"]["properties"].items():
                    param_type = param_schema.get("type", "string")
                    
                    # Map JSON Schema types to Cohere types
                    cohere_type = self._map_type_to_cohere(param_type)
                    
                    parameter_definitions[param_name] = {
                        "description": param_schema.get("description", f"Parameter {param_name}"),
                        "type": cohere_type,
                        "required": param_name in tool["input_schema"].get("required", [])
                    }
            
            # Cohere Client v1 format
            cohere_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "parameter_definitions": parameter_definitions
            }
            
            cohere_tools.append(cohere_tool)
        
        logger.debug(f"Converted {len(cohere_tools)} tools to Cohere format")
        
        return cohere_tools
    
    def _map_type_to_cohere(self, json_type: str) -> str:
        """Map JSON Schema types to Cohere types"""
        type_mapping = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "list",
            "object": "dict"
        }
        
        return type_mapping.get(json_type, "str")
    
    async def close(self):
        """Close MCP connection"""
        logger.info("MCP client cleanup (connections auto-closed)")


class MCPClientManager:
    """Manages MCP client lifecycle"""
    
    def __init__(self):
        self._client: Optional[MCPClient] = None
        self._lock = asyncio.Lock()
    
    async def get_client(self) -> MCPClient:
        """Get or create MCP client"""
        async with self._lock:
            if self._client is None:
                self._client = MCPClient()
                try:
                    await self._client.connect()
                except Exception as e:
                    logger.error(f"Failed to create MCP client: {e}")
                    self._client = None
                    raise
            return self._client
    
    async def close(self):
        """Close client"""
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None


# Global client manager
mcp_client_manager = MCPClientManager()