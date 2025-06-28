"""
Enhanced weather agent with dynamic MCP integration.
Compatible with your existing AutoGen patterns and behavior learning.
"""
import asyncio
from typing import Dict, Any, List, Optional
from autogen_core import MessageContext

from ..core.base_agent import AutoGenBaseAgent
from ...mcp.tool_discovery import YAMLToolDiscovery
from ..core.logging_config import log_structured


class ReliableWeatherAgent(AutoGenBaseAgent):
    """Weather agent with dynamic MCP tool integration and behavior learning."""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Weather information and environmental data specialist",
            agent_type="weather"
        )
        
        # Initialize MCP tool discovery
        self.tool_discovery = YAMLToolDiscovery()
        self.available_tools: List = []
        self._tools_initialized = False
    
    async def _ensure_tools_initialized(self) -> bool:
        """Ensure MCP tools are initialized."""
        if not self._tools_initialized:
            try:
                success = await self.tool_discovery.initialize()
                if success:
                    self.available_tools = await self.tool_discovery.get_tools_for_agent("weather")
                    self._tools_initialized = True
                    log_structured("weather_agent_tools_initialized",
                                  tool_count=len(self.available_tools))
                return success
            except Exception as e:
                log_structured("weather_agent_tool_init_failed", error=str(e))
                return False
        return True
    
    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process weather-related messages with dynamic MCP tool integration."""
        try:
            # Ensure tools are available
            await self._ensure_tools_initialized()
            
            log_structured("weather_processing_start",
                          message=message[:100],
                          sender=str(ctx.sender),
                          session_id=self.current_session_id)
            
            message_lower = message.lower()
            
            # Use behavior learning to analyze the query
            response = None
            if self.tool_discovery.is_behavior_learning_enabled("weather"):
                response = await self._process_with_behavior_learning(message)
            
            if not response:
                # Fallback to pattern matching
                if any(word in message_lower for word in ["weather", "temperature", "forecast", "rain", "sunny", "cloudy"]):
                    response = await self._handle_weather_query(message)
                else:
                    response = self._provide_weather_help()
            
            await self._store_interaction(message, response)
            return response
            
        except Exception as e:
            log_structured("weather_processing_error",
                          error=str(e),
                          message=message[:100],
                          agent_id=self.agent_id)
            return f"I'm having trouble with weather information right now: {str(e)}"
    
    async def _process_with_behavior_learning(self, message: str) -> Optional[str]:
        """Process using behavior learning system."""
        try:
            from ...learning.behavior.behavior_engine import analyze_query_with_llm, search_semantic_memory
            
            # Analyze query with LLM
            analysis = await analyze_query_with_llm(message)
            if not analysis:
                return None
            
            # Check if this is weather-related based on analysis
            intent_type = analysis.get("intent_type", "")
            domain = analysis.get("domain", "")
            
            if "weather" in intent_type.lower() or "weather" in domain.lower():
                # Search for relevant weather memories
                memory_results = search_semantic_memory(
                    message, 
                    n_results=3, 
                    session_id=self.current_session_id
                )
                
                if memory_results:
                    # Use memory context to inform response
                    return await self._handle_weather_query_with_memory(message, memory_results)
                else:
                    return await self._handle_weather_query(message)
            
            return None
            
        except Exception as e:
            log_structured("weather_behavior_learning_failed", error=str(e))
            return None
    
    async def _handle_weather_query(self, message: str) -> str:
        """Handle weather query using MCP tools."""
        try:
            if not self.available_tools:
                return "Weather tools are not available right now."
            
            # Find weather tool
            weather_tool = None
            for tool in self.available_tools:
                if "weather" in tool.name.lower():
                    weather_tool = tool
                    break
            
            if not weather_tool:
                return "No weather tools found."
            
            # Extract location from message (simple pattern matching)
            location = self._extract_location(message)
            
            # Determine if forecast is requested
            if any(word in message.lower() for word in ["forecast", "tomorrow", "week", "days"]):
                # Use forecast tool if available
                forecast_tool = None
                for tool in self.available_tools:
                    if "forecast" in tool.name.lower():
                        forecast_tool = tool
                        break
                
                if forecast_tool:
                    params = {"location": location, "days": 7}
                    result = await forecast_tool._arun(**params)
                    return f"Weather forecast for {location}: {result}"
            
            # Use current weather tool
            params = {"location": location, "units": "metric"}
            result = await weather_tool._arun(**params)
            return f"Current weather in {location}: {result}"
            
        except Exception as e:
            log_structured("weather_query_failed", error=str(e))
            return f"I couldn't get weather information: {str(e)}"
    
    async def _handle_weather_query_with_memory(self, message: str, memory_results: Dict) -> str:
        """Handle weather query using memory context."""
        try:
            # Extract relevant information from memory
            documents = memory_results.get("documents", [[]])[0]
            metadatas = memory_results.get("metadatas", [[]])[0]
            
            # Look for previous weather queries
            previous_locations = []
            for doc, meta in zip(documents, metadatas):
                if meta and "weather" in str(meta).lower():
                    # Extract location from previous queries
                    if "location" in str(meta):
                        previous_locations.append(meta.get("location", ""))
            
            # Use most common location as default if no location in current query
            location = self._extract_location(message)
            if not location and previous_locations:
                location = max(set(previous_locations), key=previous_locations.count)
            
            return await self._handle_weather_query(message)
            
        except Exception as e:
            log_structured("weather_memory_processing_failed", error=str(e))
            return await self._handle_weather_query(message)
    
    def _extract_location(self, message: str) -> str:
        """Extract location from message."""
        # Simple location extraction
        words = message.split()
        for i, word in enumerate(words):
            if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                return words[i + 1]
        
        # Default location
        return "Brussels"
    
    def _provide_weather_help(self) -> str:
        """Provide weather help information."""
        return """I can help you with weather information! Try asking:
        
        • "What's the weather in Brussels?"
        • "Weather forecast for tomorrow"
        • "Temperature in Paris"
        • "Will it rain today?"
        
        I can provide current conditions and forecasts for any location."""
    
    async def _store_interaction(self, message: str, response: str) -> None:
        """Store interaction with behavior learning integration."""
        try:
            # Store in database
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
                    from ...learning.behavior.behavior_engine import add_to_semantic_memory
                    await add_to_semantic_memory(
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
                log_structured("weather_interaction_stored", 
                              interaction_id=interaction_id)
                
        except Exception as e:
            log_structured("weather_interaction_storage_failed", error=str(e))
