"""
MCP Client Wrapper for Cohere
Connects to MCP server and provides tool execution in Cohere format
"""
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict, List, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP server"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
    
    async def connect(self):
        """Connect to MCP server"""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=["-m", "mcp_server.server"],
                env=None
            )
            
            logger.info("Connecting to MCP server...")
            
            # Create stdio client
            async with stdio_client(server_params) as (read, write):
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
                    
                    logger.info(f"Connected to MCP server. Available tools: {len(self.available_tools)}")
                    logger.info(f"Tools: {[t['name'] for t in self.available_tools]}")
                    
                    return self.available_tools
        
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("MCP client not connected. Call connect() first.")
        
        logger.info(f"Calling MCP tool: {tool_name} with args: {arguments}")
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            logger.info(f"Tool {tool_name} executed successfully")
            
            # Extract result content
            if result.content:
                # Get text content from first item
                return {
                    "success": True,
                    "result": result.content[0].text if result.content else None,
                    "is_error": result.isError
                }
            
            return {
                "success": True,
                "result": None,
                "is_error": result.isError
            }
        
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tools_for_cohere(self) -> List[Dict[str, Any]]:
        """
        Get tools in Cohere API format
        
        Cohere expects tools in a specific format with parameter_definitions
        
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
                        "description": param_schema.get("description", ""),
                        "type": cohere_type,
                        "required": param_name in tool["input_schema"].get("required", [])
                    }
            
            cohere_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "parameter_definitions": parameter_definitions
            }
            
            cohere_tools.append(cohere_tool)
        
        return cohere_tools
    
    def _map_type_to_cohere(self, json_type: str) -> str:
        """
        Map JSON Schema types to Cohere types
        
        Args:
            json_type: JSON Schema type (string, number, boolean, array, object)
            
        Returns:
            Cohere type (str, float, bool, list, dict)
        """
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
        if self.session:
            logger.info("Closing MCP connection")
            # Session is closed automatically with context manager


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
                await self._client.connect()
            return self._client
    
    async def close(self):
        """Close client"""
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None


# Global client manager
mcp_client_manager = MCPClientManager()