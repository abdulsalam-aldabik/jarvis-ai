"""
UPDATED: Weather agent with proper tool discovery integration
"""
import asyncio
from typing import Dict, Any, Optional, List
from autogen_core import MessageContext

from src.agents.core.base_agent import AutoGenBaseAgent
from src.agents.core.logging_config import log_structured
from src.mcp.tool_discovery import YAMLToolDiscovery


class ReliableWeatherAgent(AutoGenBaseAgent):
    """Weather agent with full tool discovery and MCP integration"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Weather information specialist with advanced MCP integration",
            agent_type="weather"
        )
        
        self.tool_discovery = YAMLToolDiscovery()
        self.weather_tools: List = []
        self.mcp_ready = False
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure tools are initialized before use"""
        if not self._initialized:
            await self._initialize_tools()
            self._initialized = True

    async def _initialize_tools(self):
        """Initialize MCP tools using YAML discovery"""
        try:
            success = await self.tool_discovery.initialize()
            if success:
                self.weather_tools = await self.tool_discovery.get_tools_for_agent("weather")
                self.mcp_ready = len(self.weather_tools) > 0
                
                log_structured("weather_agent_initialized", 
                             tools_count=len(self.weather_tools),
                             tools=[t.name for t in self.weather_tools])
            else:
                log_structured("weather_agent_init_failed")
                
        except Exception as e:
            log_structured("weather_agent_init_error", error=str(e))
            self.mcp_ready = False

    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process weather messages with full integration"""
        try:
            await self._ensure_initialized()
            
            log_structured("weather_processing_start", 
                          message=message[:100], 
                          session_id=self.current_session_id,
                          mcp_ready=self.mcp_ready)
            
            if not self._is_weather_query(message):
                return self._provide_weather_help()
            
            # Try behavior learning first
            response = await self._process_with_memory(message)
            if response:
                await self._store_interaction(message, response)
                return response
            
            # Use MCP tools
            response = await self._process_with_mcp_tools(message)
            await self._store_interaction(message, response)
            return response
            
        except Exception as e:
            log_structured("weather_processing_error", error=str(e))
            return f"⚠️ Weather service temporarily unavailable: {str(e)}"

    def _is_weather_query(self, message: str) -> bool:
        """Detect weather queries"""
        weather_keywords = [
            "weather", "temperature", "forecast", "rain", "sunny", "cloudy", 
            "hot", "cold", "humid", "wind", "storm", "snow", "climate"
        ]
        return any(keyword in message.lower() for keyword in weather_keywords)

    async def _process_with_memory(self, message: str) -> Optional[str]:
        """Use behavior learning system"""
        try:
            from src.learning.behavior.behavior_engine import analyze_query_with_llm, search_semantic_memory
            
            analysis = await analyze_query_with_llm(message)
            if not analysis or "weather" not in str(analysis).lower():
                return None
            
            memory_results = search_semantic_memory(
                message, n_results=3, session_id=self.current_session_id
            )
            
            if memory_results and memory_results.get("documents"):
                return await self._combine_memory_with_current_weather(message, memory_results)
            
            return None
            
        except ImportError:
            log_structured("behavior_learning_unavailable")
            return None
        except Exception as e:
            log_structured("behavior_learning_error", error=str(e))
            return None

    async def _process_with_mcp_tools(self, message: str) -> str:
        """Use MCP tools via tool discovery"""
        if not self.mcp_ready or not self.weather_tools:
            return self._generate_fallback_response(message)
        
        try:
            location = self._extract_location(message)
            
            weather_tool = next((tool for tool in self.weather_tools 
                               if "weather" in tool.name.lower()), None)
            
            if weather_tool:
                log_structured("weather_using_mcp_tool", 
                             tool_name=weather_tool.name, location=location)
                
                result = await weather_tool._arun(location=location)
                return self._format_weather_response(result, location)
            else:
                log_structured("weather_no_tools_available")
                return self._generate_fallback_response(message)
                
        except Exception as e:
            log_structured("mcp_tool_execution_error", error=str(e))
            return self._generate_fallback_response(message)

    async def _combine_memory_with_current_weather(self, message: str, memory_results: Dict) -> str:
        """Combine memory with current weather"""
        try:
            location = self._extract_location(message)
            current_weather = await self._get_current_weather_via_mcp(location)
            
            response_parts = [f"🌤️ **Current weather for {location}:**"]
            
            if current_weather:
                response_parts.append(current_weather)
            else:
                response_parts.append(self._get_demo_weather(location))
            
            # Add memory context
            previous_locations = self._extract_previous_locations(memory_results)
            if previous_locations:
                unique_locations = list(set(previous_locations[:3]))
                response_parts.append(f"\n💭 *I remember you've asked about: {', '.join(unique_locations)}*")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            log_structured("memory_weather_combination_error", error=str(e))
            return await self._get_current_weather_via_mcp(self._extract_location(message)) or self._generate_fallback_response(message)

    async def _get_current_weather_via_mcp(self, location: str) -> Optional[str]:
        """Get weather via MCP tools"""
        try:
            await self._ensure_initialized()
            
            if not self.weather_tools:
                return None
                
            weather_tool = next((tool for tool in self.weather_tools 
                               if "weather" in tool.name.lower()), None)
            
            if weather_tool:
                result = await weather_tool._arun(location=location)
                return self._format_json_weather(result)
            
            return None
            
        except Exception as e:
            log_structured("direct_weather_call_error", error=str(e))
            return None

    def _format_json_weather(self, weather_text: str) -> str:
        """Format weather response"""
        try:
            import json
            data = json.loads(weather_text)
            
            location = data.get("location", "Unknown")
            temp = data.get("temperature", "N/A")
            conditions = data.get("conditions", "N/A")
            humidity = data.get("humidity", "")
            
            parts = [f"📍 **{location}**", f"🌡️ {temp}", f"☁️ {conditions}"]
            if humidity:
                parts.append(f"💧 {humidity}")
            
            if data.get("demo"):
                parts.append("*(Demo mode)*")
            
            return " | ".join(parts)
            
        except json.JSONDecodeError:
            return weather_text
        except Exception:
            return weather_text

    def _extract_location(self, message: str) -> str:
        """Extract location from message"""
        words = message.split()
        
        for i, word in enumerate(words):
            if word.lower() in ["in", "for", "at", "from"] and i + 1 < len(words):
                return words[i + 1].strip("?.,!")
        
        cities = ["brussels", "etterbeek", "paris", "london", "amsterdam", "berlin"]
        for word in words:
            if word.lower() in cities:
                return word.capitalize()
        
        return "Brussels"

    def _extract_previous_locations(self, memory_results: Dict) -> List[str]:
        """Extract locations from memory"""
        try:
            metadatas = memory_results.get("metadatas", [[]])[0]
            locations = []
            
            for meta in metadatas:
                if meta and isinstance(meta, dict) and "location" in meta:
                    locations.append(meta["location"])
            
            return locations
        except:
            return []

    def _format_weather_response(self, weather_result: str, location: str) -> str:
        """Format weather response"""
        return f"🌤️ **Weather for {location}:**\n{weather_result}"

    def _generate_fallback_response(self, message: str) -> str:
        """Generate fallback response"""
        location = self._extract_location(message)
        demo_weather = self._get_demo_weather(location)
        return f"🌤️ **{location}**: {demo_weather}\n⚠️ *(Using demo data - MCP service temporarily unavailable)*"

    def _get_demo_weather(self, location: str) -> str:
        """Demo weather data"""
        demo_data = {
            "brussels": "18°C, Partly cloudy, 65% humidity",
            "etterbeek": "18°C, Mild conditions, 63% humidity", 
            "paris": "20°C, Sunny, 55% humidity",
            "london": "15°C, Rainy, 80% humidity"
        }
        return demo_data.get(location.lower(), "20°C, Mild conditions, 60% humidity")

    def _provide_weather_help(self) -> str:
        """Provide weather help"""
        return """🌤️ **I can help with weather information!** Try:
• *"What's the weather in Brussels?"*
• *"Weather forecast for tomorrow"*
• *"Temperature in Etterbeek"*
• *"Is it raining in Paris?"*

I use advanced MCP services for accurate weather data."""

    async def _store_interaction(self, message: str, response: str):
        """Store interaction in database and behavior learning"""
        try:
            interaction_id = self.database.store_agent_interaction(
                agent_id=self.agent_id,
                interaction_type="weather_query",
                data={
                    "user_message": message,
                    "agent_response": response,
                    "session_id": self.current_session_id,
                    "mcp_available": self.mcp_ready,
                    "location": self._extract_location(message)
                }
            )
            
            try:
                from src.learning.behavior.behavior_engine import add_to_semantic_memory
                await add_to_semantic_memory(
                    content=f"Weather Query: {message} | Response: {response}",
                    metadata={
                        "type": "weather_interaction",
                        "agent_id": self.agent_id,
                        "session_id": self.current_session_id,
                        "interaction_id": interaction_id,
                        "domain": "weather",
                        "location": self._extract_location(message)
                    }
                )
            except ImportError:
                pass
            
            log_structured("weather_interaction_stored", 
                         interaction_id=interaction_id)
            
        except Exception as e:
            log_structured("weather_interaction_storage_error", error=str(e))
