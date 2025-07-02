"""
YAML-based tool discovery system integrated with your infrastructure
"""
import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx

from src.agents.core.logging_config import log_structured
from src.mcp.proxy_manager import MCPProxyManager


class YAMLToolDiscovery:
    """Tool discovery system using your YAML configuration"""
    
    def __init__(self, config_path: str = "config/mcp_tools.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        self.proxy_manager = None
        self.tools_cache = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize tool discovery system"""
        try:
            # Load YAML configuration
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Initialize proxy manager with fixed timeout
            self.proxy_manager = MCPProxyManager()
            
            # Test proxy connection
            health = await self.proxy_manager.check_proxy_health()
            if health.get("status") == "healthy":
                log_structured("tool_discovery_initialized", config_path=str(self.config_path))
                self._initialized = True
                return True
            else:
                log_structured("tool_discovery_proxy_unhealthy", health=health)
                return False
                
        except Exception as e:
            log_structured("tool_discovery_init_failed", error=str(e))
            return False

    async def get_tools_for_agent(self, agent_id: str) -> List['MCPTool']:
        """Get MCP tools configured for specific agent"""
        if not self._initialized:
            await self.initialize()
            
        try:
            agent_config = self.config.get("agent_tools", {}).get(agent_id, {})
            mcp_clients = agent_config.get("mcp_clients", [])
            
            tools = []
            for client_name in mcp_clients:
                client_tools = await self.proxy_manager.get_client_tools(client_name)
                for tool_config in client_tools:
                    tool = MCPTool(
                        name=tool_config["name"],
                        description=tool_config["description"],
                        client_name=client_name,
                        proxy_manager=self.proxy_manager,
                        parameters=tool_config.get("parameters", {})
                    )
                    tools.append(tool)
            
            log_structured("agent_tools_discovered", 
                         agent_id=agent_id, tool_count=len(tools))
            return tools
            
        except Exception as e:
            log_structured("agent_tools_discovery_failed", 
                         agent_id=agent_id, error=str(e))
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Health check for tool discovery system"""
        if not self.proxy_manager:
            return {"status": "not_initialized"}
            
        proxy_health = await self.proxy_manager.check_proxy_health()
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "proxy_health": proxy_health.get("status", "unknown"),
            "config_loaded": self.config is not None,
            "initialized": self._initialized
        }

    def get_agent_specialization(self, agent_id: str) -> str:
        """Get agent specialization from YAML"""
        if not self.config:
            return "general"
        return self.config.get("agent_tools", {}).get(agent_id, {}).get("specialization", "general")

    def get_agent_a2a_capabilities(self, agent_id: str) -> List[Dict]:
        """Get A2A capabilities from YAML"""
        if not self.config:
            return []
        return self.config.get("agent_tools", {}).get(agent_id, {}).get("a2a_capabilities", [])

    def is_behavior_learning_enabled(self, agent_id: str) -> bool:
        """Check if behavior learning is enabled"""
        if not self.config:
            return False
        return self.config.get("agent_tools", {}).get(agent_id, {}).get("behavior_learning", False)

    def is_session_memory_enabled(self, agent_id: str) -> bool:
        """Check if session memory is enabled"""
        if not self.config:
            return False
        return self.config.get("agent_tools", {}).get(agent_id, {}).get("session_memory", False)

    def get_agent_class_name(self, agent_id: str) -> str:
        """Get agent class name from YAML"""
        if not self.config:
            return "Unknown"
        return self.config.get("agent_tools", {}).get(agent_id, {}).get("class_name", "Unknown")

    async def refresh_tools(self):
        """Refresh tool cache"""
        self.tools_cache.clear()
        if self.proxy_manager:
            self.proxy_manager.clear_session_cache()
        await self.initialize()


class MCPTool:
    """Individual MCP tool with proxy execution"""
    
    def __init__(self, name: str, description: str, client_name: str, proxy_manager: MCPProxyManager, parameters: Dict = None):
        self.name = name
        self.description = description
        self.client_name = client_name
        self.proxy_manager = proxy_manager
        self.parameters = parameters or {}

    async def _arun(self, **kwargs) -> str:
        """Execute tool via proxy manager"""
        try:
            result = await self.proxy_manager.call_tool(
                client_name=self.client_name,
                tool_name=self.name,
                params=kwargs
            )
            
            # Extract content from MCP response
            if isinstance(result, dict):
                if "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and content:
                        return content[0].get("text", str(result))
                return str(result)
            
            return str(result)
            
        except Exception as e:
            log_structured("mcp_tool_execution_failed", 
                         tool=self.name, client=self.client_name, error=str(e))
            return f"Tool execution failed: {str(e)}"
