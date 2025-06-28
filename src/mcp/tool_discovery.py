"""
Dynamic MCP tool discovery using YAML configuration and your tbxark/mcp-proxy setup.
Integrates with your existing AutoGen agents and behavior learning system.
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio
import time

from .proxy_manager import MCPProxyManager
from ..utils.config_loader import ConfigLoader
from ..agents.core.logging_config import log_structured

if TYPE_CHECKING:
    from ..learning.behavior.behavior_engine import add_to_semantic_memory


class YAMLMCPTool(BaseTool):
    """MCP tool discovered via YAML configuration and tbxark/mcp-proxy."""
    
    name: str
    description: str
    proxy_manager: MCPProxyManager
    client_name: str
    tool_name: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, **kwargs) -> str:
        """Synchronous execution not supported."""
        raise NotImplementedError("Use _arun for async execution")
    
    async def _arun(self, **kwargs) -> str:
        """Execute MCP tool via your tbxark/mcp-proxy with behavior learning integration."""
        try:
            log_structured("mcp_tool_execution_start",
                          tool=self.name,
                          client=self.client_name,
                          params=kwargs)
            
            # Call tool via proxy
            result = await self.proxy_manager.call_tool(
                self.client_name,
                self.tool_name,
                kwargs
            )
            
            if "error" in result:
                error_response = f"Tool execution failed: {result['error']}"
                
                # Store error in behavior learning system
                try:
                    from ..learning.behavior.behavior_engine import add_to_semantic_memory
                    await add_to_semantic_memory(
                        content=f"Tool error: {self.name} - {error_response}",
                        metadata={
                            "type": "tool_error",
                            "tool_name": self.name,
                            "client_name": self.client_name,
                            "error": result['error']
                        }
                    )
                except ImportError:
                    pass
                
                return error_response
            
            # Extract content from various possible response formats
            response_content = self._extract_response_content(result)
            
            # Store successful tool use in behavior learning system
            try:
                from ..learning.behavior.behavior_engine import add_to_semantic_memory
                await add_to_semantic_memory(
                    content=f"Tool success: {self.name} - {response_content[:200]}",
                    metadata={
                        "type": "tool_success",
                        "tool_name": self.name,
                        "client_name": self.client_name,
                        "params_used": kwargs
                    }
                )
            except ImportError:
                pass
            
            log_structured("mcp_tool_execution_success",
                          tool=self.name,
                          response_length=len(response_content))
            
            return response_content
            
        except Exception as e:
            error_msg = f"Tool execution error for {self.name}: {str(e)}"
            log_structured("mcp_tool_execution_error",
                          tool=self.name,
                          error=str(e))
            return error_msg
    
    def _extract_response_content(self, result: Any) -> str:
        """Extract meaningful content from MCP response."""
        if isinstance(result, dict):
            # Try common response keys
            for key in ["content", "result", "response", "data", "output"]:
                if key in result and result[key]:
                    return str(result[key])
            
            # If no standard key, return JSON representation
            return str(result)
        
        return str(result)


class YAMLToolDiscovery:
    """
    Discovers and creates tools based on YAML configuration.
    Integrates with your existing AutoGen agents and behavior learning.
    """
    
    def __init__(self, config_path: str = "config/mcp_tools.yaml"):
        self.config = ConfigLoader.load_yaml(config_path)
        self.proxy_manager = MCPProxyManager(config_path)
        self._tool_cache: Dict[str, List[YAMLMCPTool]] = {}
        self._agent_tool_cache: Dict[str, List[YAMLMCPTool]] = {}
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize tool discovery with your MCP proxy."""
        try:
            # Check if your proxy is running
            health = await self.proxy_manager.check_proxy_health()
            if not health["proxy_running"]:
                log_structured("mcp_proxy_not_running", port=9190)
                return False
            
            # Discover all tools from YAML configuration
            await self._discover_yaml_tools()
            
            # Create agent-specific tool caches
            await self._create_agent_tool_caches()
            
            self._initialized = True
            
            total_tools = sum(len(tools) for tools in self._tool_cache.values())
            log_structured("yaml_tool_discovery_initialized", 
                          total_tools=total_tools,
                          clients=list(self._tool_cache.keys()))
            return True
            
        except Exception as e:
            log_structured("tool_discovery_initialization_failed", error=str(e))
            return False
    
    async def _discover_yaml_tools(self) -> None:
        """Discover tools from YAML configuration."""
        clients_config = self.config.get("mcp_proxy", {}).get("clients", {})
        
        for client_name, client_config in clients_config.items():
            try:
                tools = await self._create_tools_from_yaml(client_name, client_config)
                self._tool_cache[client_name] = tools
                log_structured("yaml_client_tools_discovered",
                              client=client_name,
                              tool_count=len(tools))
                
            except Exception as e:
                log_structured("yaml_client_discovery_failed",
                              client=client_name,
                              error=str(e))
                self._tool_cache[client_name] = []
    
    async def _create_tools_from_yaml(
        self, 
        client_name: str, 
        client_config: Dict[str, Any]
    ) -> List[YAMLMCPTool]:
        """Create tools from YAML capabilities definition."""
        tools = []
        capabilities = client_config.get("capabilities", [])
        
        for capability in capabilities:
            tool_name = capability.get("name")
            if not tool_name:
                continue
                
            tool = YAMLMCPTool(
                name=f"{client_name}_{tool_name}",
                description=capability.get("description", f"{tool_name} via {client_name}"),
                proxy_manager=self.proxy_manager,
                client_name=client_name,
                tool_name=tool_name,
                parameters_schema=capability.get("parameters", {})
            )
            tools.append(tool)
        
        return tools
    
    async def _create_agent_tool_caches(self) -> None:
        """Create agent-specific tool caches based on YAML configuration."""
        agent_tools_config = self.config.get("agent_tools", {})
        
        for agent_id, agent_config in agent_tools_config.items():
            mcp_clients = agent_config.get("mcp_clients", [])
            agent_tools = []
            
            for client_name in mcp_clients:
                if client_name in self._tool_cache:
                    agent_tools.extend(self._tool_cache[client_name])
            
            self._agent_tool_cache[agent_id] = agent_tools
            log_structured("agent_tool_cache_created",
                          agent_id=agent_id,
                          tool_count=len(agent_tools),
                          clients=mcp_clients)
    
    async def get_tools_for_agent(self, agent_id: str) -> List[YAMLMCPTool]:
        """Get tools configured for a specific agent in YAML (compatible with existing agents)."""
        if not self._initialized:
            await self.initialize()
        
        # Return cached tools for agent
        agent_tools = self._agent_tool_cache.get(agent_id, [])
        
        log_structured("agent_tools_retrieved",
                      agent_id=agent_id,
                      tool_count=len(agent_tools))
        return agent_tools
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get complete agent configuration from YAML."""
        return self.config.get("agent_tools", {}).get(agent_id, {})
    
    def get_agent_a2a_capabilities(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get A2A capabilities for an agent from YAML."""
        agent_config = self.get_agent_config(agent_id)
        return agent_config.get("a2a_capabilities", [])
    
    def get_agent_specialization(self, agent_id: str) -> str:
        """Get agent specialization from YAML."""
        agent_config = self.get_agent_config(agent_id)
        return agent_config.get("specialization", "General purpose agent")
    
    def get_agent_class_name(self, agent_id: str) -> str:
        """Get agent class name from YAML."""
        agent_config = self.get_agent_config(agent_id)
        return agent_config.get("class_name", "AutoGenBaseAgent")
    
    def is_behavior_learning_enabled(self, agent_id: str) -> bool:
        """Check if behavior learning is enabled for agent."""
        agent_config = self.get_agent_config(agent_id)
        return agent_config.get("behavior_learning", False)
    
    def is_session_memory_enabled(self, agent_id: str) -> bool:
        """Check if session memory is enabled for agent."""
        agent_config = self.get_agent_config(agent_id)
        return agent_config.get("session_memory", False)
    
    async def get_all_tools(self) -> List[YAMLMCPTool]:
        """Get all discovered tools."""
        if not self._initialized:
            await self.initialize()
        
        all_tools = []
        for tools in self._tool_cache.values():
            all_tools.extend(tools)
        return all_tools
    
    async def refresh_tools(self) -> None:
        """Refresh tool discovery."""
        self._tool_cache.clear()
        self._agent_tool_cache.clear()
        await self._discover_yaml_tools()
        await self._create_agent_tool_caches()
        log_structured("tool_discovery_refreshed")
    
    async def get_proxy_status(self) -> Dict[str, Any]:
        """Get comprehensive proxy status with tool information."""
        health = await self.proxy_manager.check_proxy_health()
        capabilities = await self.proxy_manager.discover_all_capabilities()
        
        return {
            **health,
            "capabilities": capabilities,
            "yaml_tools_cached": len(self._tool_cache),
            "agent_caches": len(self._agent_tool_cache),
            "total_tools": sum(len(tools) for tools in self._tool_cache.values()),
            "behavior_integration": self.config.get("behavior_integration", {}).get("enabled", False)
        }
