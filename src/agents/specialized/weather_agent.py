"""
FIXED: Enhanced weather agent with dynamic MCP integration.
Compatible with your existing AutoGen patterns and behavior learning.
"""
import asyncio
from typing import Dict, Any, List, Optional
from autogen_core import MessageContext

from src.agents.core.base_agent import AutoGenBaseAgent
from src.mcp.tool_discovery import YAMLToolDiscovery
from src.agents.core.logging_config import log_structured


class ReliableWeatherAgent(AutoGenBaseAgent):
    """Weather agent with dynamic MCP tool integration and behavior learning."""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Weather information and environmental data specialist",
            agent_type="weather"
        )
        
        # Initialize YAMLToolDiscovery for MCP integration
        self.tool_discovery = YAMLToolDiscovery()
        self.available_tools: List = []
        self._tools_initialized = False

    def _ensure_tools_initialized(self) -> bool:  # Remove async
        """Ensure MCP tools are initialized via YAML configuration."""
        if not self._tools_initialized:
            try:
                # FIXED: No await since initialize() is now synchronous
                success = self.tool_discovery.initialize()
                
                if success:
                    # This can stay async since get_tools_for_agent might be async
                    import asyncio
                    self.available_tools = asyncio.run(
                        self.tool_discovery.get_tools_for_agent("weather")
                    )
                    self._tools_initialized = True
                    log_structured("weather_agent_tools_initialized", 
                                tool_count=len(self.available_tools))
                else:
                    log_structured("weather_agent_tool_init_failed", 
                                reason="tool_discovery_initialization_failed")
                return success
            except Exception as e:
                log_structured("weather_agent_tool_init_failed", error=str(e))
                return False
        return True


    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process weather-related messages with dynamic MCP tool integration."""
        try:
            log_structured("weather_processing_start", 
                          message=message[:100], 
                          sender=str(ctx.sender), 
                          session_id=self.current_session_id)
            
            # Initialize MCP tool discovery
            self._ensure_tools_initialized()
            
            message_lower = message.lower()
            
            # Process with behavior learning first
            response = None
            if self.tool_discovery.is_behavior_learning_enabled("weather"):
                response = await self._process_with_behavior_learning(message)
            
            if not response:
                # Use behavior learning to analyze the query
                if any(word in message_lower for word in 
                      ["weather", "temperature", "forecast", "rain", "sunny", "cloudy"]):
                    response = await self._handle_weather_query(message)
                else:
                    response = self._provide_weather_help()
            
            await self._store_interaction(message, response)
            return response
            
        except Exception as e:
            log_structured("weather_processing_error", 
                          error=str(e), message=message[:100], 
                          agent_id=self.agent_id)
            return f"I'm having trouble with weather information right now: {str(e)}"

    async def _process_with_behavior_learning(self, message: str) -> Optional[str]:
        """Process using behavior learning system."""
        try:
            from src.learning.behavior.behavior_engine import analyze_query_with_llm, search_semantic_memory
            
            # Analyze query with LLM
            analysis = await analyze_query_with_llm(message)
            if not analysis:
                return None
            
            # Check if weather-related
            intent_type = analysis.get("intent_type", "")
            domain = analysis.get("domain", "")
            
            if any(term in str(val).lower() for val in [intent_type, domain] 
                   for term in ["weather", "environmental", "meteorological"]):
                # Check for relevant weather memories
                memory_results = search_semantic_memory(
                    message, n_results=3, session_id=self.current_session_id
                )
                if memory_results:
                    return await self._handle_weather_query_with_memory(message, memory_results)
                else:
                    return await self._handle_weather_query(message)
            
            return None
            
        except Exception as e:
            log_structured("weather_behavior_learning_failed", error=str(e))
            return None

    async def _handle_weather_query(self, message: str) -> str:
        """Handle weather query with direct MCP fallback."""
        try:
            if not self._tools_initialized:
                self._ensure_tools_initialized()  # Remove await!
            
            location = self._extract_location(message)
            
            # Try available tools first
            if self.available_tools:
                for tool in self.available_tools:
                    if "weather" in tool.name.lower():
                        try:
                            result = await tool._arun(location=location)
                            return f"Current weather in {location}: {result}"
                        except Exception as e:
                            log_structured("tool_execution_failed", error=str(e))
            
            # Fallback to direct MCP proxy call
            result = await self._call_mcp_tool_directly("get_weather", {"location": location})
            return f"Current weather in {location}: {result}"
            
        except Exception as e:
            log_structured("weather_query_failed", error=str(e))
            return "Weather service is temporarily unavailable."



    async def _handle_weather_query_with_memory(self, message: str, memory_results: Dict) -> str:
        """Handle weather query using memory context."""
        try:
            # Extract relevant information from memory
            documents = memory_results.get("documents", [[]])[0]
            metadatas = memory_results.get("metadatas", [[]])[0]
            
            # Look for previous locations in memory
            previous_locations = []
            for doc, meta in zip(documents, metadatas):
                if meta and "weather" in str(meta).lower():
                    if "location" in str(meta):
                        previous_locations.append(meta.get("location", ""))
            
            # Use current weather tools with memory context
            return await self._handle_weather_query(message)
            
        except Exception as e:
            log_structured("weather_memory_processing_failed", error=str(e))
            return await self._handle_weather_query(message)

    def _extract_location(self, message: str) -> str:
        """Extract location from message."""
        words = message.split()
        
        # Look for location patterns
        for i, word in enumerate(words):
            if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                return words[i + 1].strip("?.,!")
        
        # Look for common cities
        cities = ["brussels", "paris", "london", "berlin", "amsterdam", "rome", "madrid", "vienna"]
        for word in words:
            if word.lower() in cities:
                return word.capitalize()
        
        return "Brussels"  # Default

    def _extract_days(self, message: str) -> int:
        """Extract number of forecast days."""
        if "tomorrow" in message.lower():
            return 1
        elif "week" in message.lower():
            return 7
        elif any(word in message.lower() for word in ["3", "three"]):
            return 3
        elif any(word in message.lower() for word in ["5", "five"]):
            return 5
        return 3  # Default

    def _provide_weather_help(self) -> str:
        """Provide weather help information."""
        return """I can help you with weather information! Try asking:
• "What's the weather in Brussels?"
• "Weather forecast for tomorrow"
• "Temperature in Paris"
• "Will it rain today?"

I can provide current conditions and forecasts for any location."""

    async def _store_interaction(self, message: str, response: str) -> None:
        """FIXED: Store interaction - removed incorrect await."""
        try:
            # FIXED: Don't await this synchronous method!
            interaction_id = self.database.store_agent_interaction(
                agent_id=self.agent_id,
                interaction_type="weather_query",
                data={
                    "user_message": message,
                    "agent_response": response,
                    "session_id": self.current_session_id,
                    "tools_available": len(self.available_tools)
                }
            )
            
            # Store in behavior learning system if enabled
            if self.tool_discovery.is_behavior_learning_enabled("weather"):
                try:
                    from src.learning.behavior.behavior_engine import add_to_semantic_memory
                    add_to_semantic_memory(
                        content=f"Weather Query: {message} | Response: {response}",
                        metadata={
                            "type": "weather_interaction",
                            "agent_id": self.agent_id,
                            "session_id": self.current_session_id,
                            "interaction_id": interaction_id,
                            "domain": "weather",
                            "intent_type": "weather_query"
                        }
                    )
                except ImportError:
                    pass
            
            if interaction_id:
                log_structured("weather_interaction_stored", interaction_id=interaction_id)
                
        except Exception as e:
            log_structured("weather_interaction_storage_failed", error=str(e))


    async def _call_mcp_tool_directly(self, tool_name: str, params: dict) -> str:
        """Direct MCP proxy call with proper sessionId."""
        try:
            import httpx
            import time
            
            # Use timestamp-based session ID
            session_id = f"session_{int(time.time())}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.tool_discovery.proxy_manager.base_url}/weather/message?sessionId={session_id}",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": params
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result:
                        return str(result["result"])
                    else:
                        return str(result)
                else:
                    log_structured("direct_mcp_call_failed", 
                                status_code=response.status_code,
                                response=response.text)
                    return f"Weather service error: {response.status_code}"
                    
        except Exception as e:
            log_structured("direct_mcp_call_exception", error=str(e))
            return f"Weather service temporarily unavailable: {str(e)}"
