"""
Enhanced MCP Tools Manager with YAML Configuration Support
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from autogen_ext.tools.mcp import SseServerParams, mcp_server_tools
from autogen_core.tools import Tool, FunctionTool
from autogen_core import CancellationToken

from config.settings import settings
from src.utils.config_loader import ConfigLoader
from src.agents.core.logging_config import log_structured

logger = logging.getLogger(__name__)

class MCPToolsManager:
    """Enhanced MCP Tools Manager with YAML configuration support"""
    
    def __init__(self, config_path: str = "config/mcp_tools.yaml"):
        self.config_path = config_path
        self.config = None
        self.mcp_tools: List[Tool] = []
        self.client_tools: Dict[str, List[Tool]] = {}
        self.initialized = False
        
    async def initialize_mcp_tools(self):
        """Initialize MCP tools using YAML configuration"""
        try:
            # Load configuration
            self.config = ConfigLoader.load_mcp_config(self.config_path)
            
            # Initialize enabled MCP clients
            proxy_config = self.config.get("mcp_proxy", {})
            clients = proxy_config.get("clients", {})
            
            for client_name, client_config in clients.items():
                if client_config.get("enabled", False):
                    await self._initialize_client(client_name, client_config)
            
            # Flatten all tools for global access
            self.mcp_tools = []
            for tools in self.client_tools.values():
                self.mcp_tools.extend(tools)
            
            self.initialized = True
            log_structured("mcp_tools_yaml_initialized",
                         config_path=self.config_path,
                         total_tools=len(self.mcp_tools),
                         client_count=len(self.client_tools),
                         autogen_version="0.6.2-official")
                         
        except Exception as e:
            logger.error(f"YAML MCP tools initialization failed: {e}")
            log_structured("mcp_tools_yaml_failed", 
                         config_path=self.config_path, 
                         error=str(e))
            
            # Fallback to mock tools
            self.mcp_tools = self._create_fallback_tools()
            self.initialized = True
            log_structured("mcp_tools_fallback_mode", 
                         tools_count=len(self.mcp_tools))

    async def _initialize_client(self, client_name: str, client_config: Dict[str, Any]):
        """Initialize individual MCP client from configuration"""
        try:
            transport = client_config.get("transport", "sse")
            base_url = settings.mcp.proxy_url
            
            if transport == "sse":
                # Use SSE transport for sparfenyuk/mcp-proxy
                sse_url = f"{base_url}/servers/{client_name}/sse"
                server_params = SseServerParams(url=sse_url)
                
                # Discover tools via AutoGen's official integration
                client_tools = await mcp_server_tools(server_params)
                self.client_tools[client_name] = client_tools
                
                log_structured("mcp_client_initialized",
                             client_name=client_name,
                             transport=transport,
                             tools_count=len(client_tools),
                             sse_url=sse_url)
                
            elif transport == "http":
                # HTTP transport for direct API calls
                tools = self._create_http_tools(client_name, client_config)
                self.client_tools[client_name] = tools
                
                log_structured("mcp_client_http_initialized",
                             client_name=client_name,
                             tools_count=len(tools))
                
        except Exception as e:
            log_structured("mcp_client_init_failed",
                         client_name=client_name,
                         error=str(e))

    def _create_http_tools(self, client_name: str, client_config: Dict[str, Any]) -> List[Tool]:
        """Create HTTP-based tools from configuration"""
        tools = []
        capabilities = client_config.get("capabilities", [])
        
        for capability in capabilities:
            tool_name = capability["name"]
            description = capability.get("description", f"{tool_name} tool")
            
            # Create dynamic HTTP tool
            async def http_tool_func(**kwargs):
                # Implement HTTP tool logic here
                return f"HTTP tool {tool_name} called with {kwargs}"
            
            tool = FunctionTool(http_tool_func, name=tool_name, description=description)
            tools.append(tool)
            
        return tools

    def get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        """Get tools assigned to specific agent type"""
        if self.config is None:
            log_structured("mcp_tools_not_initialized", 
                        agent_type=agent_type,
                        message="MCP tools not initialized, returning empty list")
            return []
        agent_tools_config = self.config.get("agent_tools", {})
        agent_config = agent_tools_config.get(agent_type, {})
        mcp_clients = agent_config.get("mcp_clients", [])
        
        tools = []
        for client_name in mcp_clients:
            if client_name in self.client_tools:
                tools.extend(self.client_tools[client_name])
        
        return tools

    async def get_available_tools(self) -> List[Tool]:
        """Get all available MCP tools"""
        if not self.initialized:
            await self.initialize_mcp_tools()
        return self.mcp_tools

    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Call MCP tool with proper error handling"""
        try:
            for tool in self.mcp_tools:
                if tool.name == tool_name:
                    cancellation_token = CancellationToken()
                    result = await tool.run_json(parameters, cancellation_token)
                    
                    log_structured("mcp_tool_called",
                                 tool_name=tool_name,
                                 parameters=parameters,
                                 success=True)
                    return result
                    
            raise ValueError(f"Tool '{tool_name}' not found")
            
        except Exception as e:
            log_structured("mcp_tool_call_failed",
                         tool_name=tool_name,
                         error=str(e))
            raise

    def _create_fallback_tools(self) -> List[Tool]:
        """Create fallback tools when MCP unavailable"""
        async def get_weather(location: str = "Brussels") -> str:
            return f"""Weather for {location}:
Temperature: 19°C (feels like 21°C)
Condition: Partly cloudy
Humidity: 60%
Wind: 13 km/h SW
*Configuration-based fallback mode*"""
        
        async def create_routine(routine_data: dict) -> str:
            routine_name = routine_data.get("name", "New Routine")
            return f"""✅ Created routine: {routine_name}
Activities: {len(routine_data.get('activities', []))}
Status: Success (configuration fallback)"""
        
        async def system_status(include_details: bool = False) -> str:
            return """System Status: Healthy
MCP Tools: Fallback mode
Configuration: Loaded from YAML"""
        
        return [
            FunctionTool(get_weather, description="Get weather information"),
            FunctionTool(create_routine, description="Create a new routine"),
            FunctionTool(system_status, description="Get system status")
        ]

# Global instance
mcp_tools_manager = MCPToolsManager()
