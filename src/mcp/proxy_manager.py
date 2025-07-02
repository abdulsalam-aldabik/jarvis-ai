"""
Updated MCP proxy manager using mcp_tools.yaml configuration
"""
import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx

from src.utils.config_loader import ConfigLoader
from src.agents.core.logging_config import log_structured


class MCPProxyManager:
    """Updated proxy manager using mcp_tools.yaml config"""
    
    def __init__(self, config_path: str = "config/mcp_tools.yaml"):
        self.config_path = Path(config_path)
        self.config = ConfigLoader.load_yaml(str(self.config_path))
        self.session_cache = {}
        self.max_retries = 2
        
        # Get proxy configuration from mcp_tools.yaml
        proxy_config = self.config.get("mcp_proxy", {}).get("server", {})
        host = proxy_config.get("host", "mcp-proxy")
        port = proxy_config.get("port", 9190)
        
        self.base_url = f"http://{host}:{port}"
        self.transport_config = self.config.get("proxy_transport_config", {})
        
        log_structured("mcp_proxy_manager_initialized", 
                      base_url=self.base_url, port=port)

    def get_client_transport_config(self, client_name: str) -> Dict[str, Any]:
        """Get transport configuration for a client"""
        return self.transport_config.get(client_name, {
            "type": "stdio_via_sse",
            "sse_endpoint": f"/servers/{client_name}/sse",
            "messages_endpoint": f"/servers/{client_name}/messages",
            "session_management": "auto"
        })

    async def get_session_for_client(self, client_name: str) -> str:
        """Get session ID for client using SSE endpoint"""
        if client_name in self.session_cache:
            return self.session_cache[client_name]
        
        transport_config = self.get_client_transport_config(client_name)
        sse_endpoint = f"{self.base_url}{transport_config['sse_endpoint']}"
        
        try:
            timeout = httpx.Timeout(10.0)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", sse_endpoint) as response:
                    if response.status_code == 200:
                        buffer = ""
                        async for chunk in response.aiter_text():
                            buffer += chunk
                            
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                
                                if line.startswith('data: ') and 'session_id=' in line:
                                    session_match = re.search(r'session_id=([a-f0-9]{32})', line)
                                    if session_match:
                                        session_id = session_match.group(1)
                                        self.session_cache[client_name] = session_id
                                        log_structured("mcp_session_extracted", 
                                                     client=client_name, session_id=session_id)
                                        return session_id
        except Exception as e:
            log_structured("mcp_session_extraction_failed", client=client_name, error=str(e))
        
        # Fallback
        fallback_id = str(uuid.uuid4()).replace('-', '')
        log_structured("mcp_session_fallback", client=client_name, session_id=fallback_id)
        return fallback_id

    async def call_tool(self, client_name: str, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool with complete SSE response handling"""
        
        for attempt in range(self.max_retries):
            try:
                # Get session and transport config
                session_id = await self.get_session_for_client(client_name)
                transport_config = self.get_client_transport_config(client_name)
                
                # Build endpoint
                messages_endpoint = transport_config['messages_endpoint']
                endpoint = f"{self.base_url}{messages_endpoint}/?session_id={session_id}"
                
                # Prepare request
                mcp_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": params
                    }
                }
                
                log_structured("mcp_tool_call_attempt", 
                            client=client_name, tool=tool_name, 
                            endpoint=endpoint, session_id=session_id, attempt=attempt+1)
                
                # Make request
                timeout = httpx.Timeout(30.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(endpoint, json=mcp_request)
                    
                    if response.status_code == 200:
                        # Synchronous response
                        result = response.json()
                        log_structured("mcp_tool_call_success", 
                                    client=client_name, tool=tool_name)
                        
                        if "result" in result and "content" in result["result"]:
                            content = result["result"]["content"]
                            if isinstance(content, list) and content and "text" in content[0]:
                                return {"content": content[0]["text"]}
                        return result.get("result", result)
                    
                    elif response.status_code == 202:
                        # Async response - wait for actual result
                        log_structured("mcp_async_response_wait", 
                                    client=client_name, tool=tool_name, session_id=session_id)
                        
                        # Wait for the actual response via SSE
                        actual_response = await self.wait_for_tool_response(client_name, session_id, request_id=1)
                        if actual_response:
                            return actual_response
                        else:
                            return {"error": "No response received within timeout"}
                    
                    elif response.status_code == 404:
                        # Clear session and retry
                        if client_name in self.session_cache:
                            del self.session_cache[client_name]
                        continue
                    
                    else:
                        return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
                        
            except Exception as e:
                log_structured("mcp_tool_call_exception", 
                            client=client_name, tool=tool_name, 
                            attempt=attempt+1, error=str(e))
                continue
        
        return {"error": f"Tool call failed after {self.max_retries} attempts"}

    async def wait_for_tool_response(self, client_name: str, session_id: str, request_id: int, timeout_seconds: int = 10) -> Optional[Dict[str, Any]]:
        """Wait for tool response via SSE stream"""
        try:
            transport_config = self.get_client_transport_config(client_name)
            sse_endpoint = f"{self.base_url}{transport_config['sse_endpoint']}"
            
            log_structured("mcp_sse_response_listen", 
                        client=client_name, session_id=session_id, 
                        endpoint=sse_endpoint, request_id=request_id)
            
            timeout = httpx.Timeout(timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", sse_endpoint) as response:
                    if response.status_code == 200:
                        buffer = ""
                        start_time = asyncio.get_event_loop().time()
                        
                        async for chunk in response.aiter_text():
                            # Check timeout
                            if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                                log_structured("mcp_sse_response_timeout", 
                                            client=client_name, session_id=session_id)
                                break
                            
                            buffer += chunk
                            
                            # Process complete lines
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                
                                # Look for JSON response data
                                if line.startswith('data: {'):
                                    try:
                                        json_data = line[6:]  # Remove 'data: '
                                        result = json.loads(json_data)
                                        
                                        log_structured("mcp_sse_json_received", 
                                                    client=client_name, session_id=session_id,
                                                    response_preview=str(result)[:200])
                                        
                                        # Check if this is our response (matching request ID)
                                        if result.get("id") == request_id and "result" in result:
                                            log_structured("mcp_tool_response_received", 
                                                        client=client_name, tool_name="get_weather")
                                            
                                            # Extract the actual weather data
                                            if "content" in result["result"]:
                                                content = result["result"]["content"]
                                                if isinstance(content, list) and content and "text" in content[0]:
                                                    return {"content": content[0]["text"]}
                                            return result["result"]
                                        
                                        # Also handle error responses
                                        elif result.get("id") == request_id and "error" in result:
                                            return {"error": f"MCP error: {result['error']}"}
                                        
                                    except json.JSONDecodeError as e:
                                        log_structured("mcp_sse_json_parse_error", 
                                                    client=client_name, line=line[:100], error=str(e))
                                        continue
            
            log_structured("mcp_sse_no_response", client=client_name, session_id=session_id)
            return None
            
        except Exception as e:
            log_structured("mcp_sse_response_error", 
                        client=client_name, session_id=session_id, error=str(e))
            return None


    async def check_proxy_health(self) -> Dict[str, Any]:
        """Check proxy health using config"""
        try:
            health_endpoint = self.config.get("proxy_config", {}).get("health_check", {}).get("endpoint", "/status")
            
            timeout = httpx.Timeout(10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.base_url}{health_endpoint}")
                
                return {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "proxy_running": True,
                    "base_url": self.base_url,
                    "response_code": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "proxy_running": False,
                "base_url": self.base_url,
                "error": str(e)
            }

    async def get_client_tools(self, client_name: str) -> List[Dict[str, Any]]:
        """Get tools from mcp_tools.yaml configuration"""
        try:
            clients_config = self.config.get("mcp_proxy", {}).get("clients", {})
            client_config = clients_config.get(client_name, {})
            capabilities = client_config.get("capabilities", [])
            
            log_structured("mcp_client_tools_retrieved", 
                         client=client_name, tool_count=len(capabilities))
            return capabilities
                
        except Exception as e:
            log_structured("mcp_client_tools_failed", 
                          client=client_name, error=str(e))
            return []

    def clear_session_cache(self):
        """Clear all cached sessions"""
        self.session_cache.clear()
        log_structured("mcp_session_cache_cleared")
