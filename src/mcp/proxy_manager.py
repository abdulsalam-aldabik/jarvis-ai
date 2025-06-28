"""
tbxark/mcp-proxy integration manager for your Jarvis system.
Integrates with your existing AutoGen + LangGraph + Behavior Learning architecture.
"""
import asyncio
import json
from pathlib import Path  
from typing import Dict, Any, List, Optional
import httpx
import time

from ..utils.config_loader import ConfigLoader
from ..agents.core.logging_config import log_structured


class MCPProxyManager:
    """Manages tbxark/mcp-proxy with your custom Docker setup on port 9190."""
    
    def __init__(self, config_path: str = "config/mcp_tools.yaml"):
        self.config_path = Path(config_path)
        self.config = ConfigLoader.load_yaml(str(self.config_path))
        
        # Use your custom port 9190 from Docker setup
        proxy_server = self.config["mcp_proxy"]["server"]
        self.base_url = f"http://{proxy_server['host']}:{proxy_server['port']}"
        
        log_structured("mcp_proxy_manager_initialized", 
                      base_url=self.base_url, 
                      port=proxy_server['port'])
        
    async def check_proxy_health(self) -> Dict[str, Any]:
        """Check if your custom tbxark/mcp-proxy is running on port 9190."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    health_data = response.json()
                    return {
                        "status": "healthy",
                        "proxy_running": True,
                        "base_url": self.base_url,
                        "port": 9190,
                        "docker_setup": "custom_tbxark",
                        **health_data
                    }
        except Exception as e:
            log_structured("mcp_proxy_health_check_failed", error=str(e))
        
        return {
            "status": "unhealthy", 
            "proxy_running": False,
            "base_url": self.base_url,
            "error": "tbxark/mcp-proxy not responding on port 9190"
        }
    
    async def get_available_clients(self) -> List[str]:
        """Get list of available MCP clients from your proxy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/mcp/list", timeout=10.0)
                response.raise_for_status()
                
                data = response.json()
                if isinstance(data, dict):
                    if "servers" in data:
                        return list(data["servers"].keys())
                    elif "clients" in data:
                        return list(data["clients"].keys())
                elif isinstance(data, list):
                    return [item.get("name", str(item)) for item in data]
                else:
                    return []
                    
        except Exception as e:
            log_structured("mcp_clients_discovery_failed", error=str(e))
            return []
    
    async def get_client_tools(self, client_name: str) -> List[Dict[str, Any]]:
        """Get tools available from a specific MCP client."""
        try:
            async with httpx.AsyncClient() as client:
                # Try multiple possible endpoints
                endpoints = [
                    f"/mcp/{client_name}/tools",
                    f"/clients/{client_name}/tools",
                    f"/mcp/servers/{client_name}/tools"
                ]
                
                for endpoint in endpoints:
                    try:
                        response = await client.get(
                            f"{self.base_url}{endpoint}",
                            timeout=10.0
                        )
                        if response.status_code == 200:
                            data = response.json()
                            return data.get("tools", []) if isinstance(data, dict) else []
                    except:
                        continue
                
                return []
                
        except Exception as e:
            log_structured("mcp_client_tools_failed", 
                          client=client_name, 
                          error=str(e))
            return []
    
    async def call_tool(
        self, 
        client_name: str, 
        tool_name: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool via your tbxark/mcp-proxy setup.""" 
        try:
            async with httpx.AsyncClient() as client:
                # Try multiple possible endpoints
                endpoints = [
                    f"/mcp/{client_name}/call",
                    f"/clients/{client_name}/call",
                    f"/mcp/servers/{client_name}/call"
                ]
                
                for endpoint in endpoints:
                    try:
                        response = await client.post(
                            f"{self.base_url}{endpoint}",
                            json={
                                "method": tool_name,
                                "params": params
                            },
                            timeout=30.0
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            log_structured("mcp_tool_call_success",
                                         client=client_name,
                                         tool=tool_name,
                                         endpoint=endpoint)
                            return result
                    except:
                        continue
                
                # If all endpoints fail
                return {"error": f"All endpoints failed for {client_name}.{tool_name}"}
                
        except httpx.TimeoutException:
            error_msg = f"Tool {tool_name} on {client_name} timed out"
            log_structured("mcp_tool_timeout", 
                          client=client_name, 
                          tool=tool_name)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Tool {tool_name} on {client_name} failed: {str(e)}"
            log_structured("mcp_tool_call_failed", 
                          client=client_name, 
                          tool=tool_name, 
                          error=str(e))
            return {"error": error_msg}
    
    def get_yaml_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get agent configuration from YAML (compatible with your existing agents)."""
        agent_tools = self.config.get("agent_tools", {})
        return agent_tools.get(agent_id, {})
    
    def get_yaml_client_config(self, client_name: str) -> Dict[str, Any]:
        """Get client configuration from YAML."""
        clients_config = self.config.get("mcp_proxy", {}).get("clients", {})
        return clients_config.get(client_name, {})
    
    async def discover_all_capabilities(self) -> Dict[str, Any]:
        """Discover all capabilities from your MCP proxy setup."""
        try:
            capabilities = {}
            clients = await self.get_available_clients()
            
            for client_name in clients:
                tools = await self.get_client_tools(client_name)
                capabilities[client_name] = {
                    "tools": tools,
                    "tool_count": len(tools),
                    "available": len(tools) > 0,
                    "yaml_config": self.get_yaml_client_config(client_name)
                }
            
            return {
                "proxy_url": self.base_url,
                "client_count": len(clients),
                "clients": capabilities,
                "discovery_timestamp": time.time(),
                "integration": "tbxark_mcp_proxy_custom"
            }
            
        except Exception as e:
            log_structured("mcp_capability_discovery_failed", error=str(e))
            return {"error": str(e)}
