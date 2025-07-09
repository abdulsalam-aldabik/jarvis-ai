"""
AutoGen 0.6.2 Weather AssistantAgent with MCP Tools
Following: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
"""
import os
import asyncio
import time
from typing import Dict, Any, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.tools.mcp import mcp_server_tools

from src.agents.core.base_agent import AgentBase, get_model_client
from src.agents.core.mcp_tools import mcp_tools_manager
from src.agents.core.logging_config import log_structured
from src.follow_up_generator import follow_up_generator

class WeatherAgent(AgentBase):
    """CORRECT AutoGen 0.6.2 Weather AssistantAgent with MCP tools integration"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="AccuWeather-powered weather agent with AutoGen 0.6.2 and MCP tools",
            agent_type="weather",
            system_message="""You are a weather specialist using AutoGen 0.6.2 architecture. 
            You provide accurate, helpful weather information and forecasts using MCP tools.
            
            Guidelines:
            - Provide clear, actionable weather information
            - Include relevant details like temperature, conditions, and recommendations
            - Use conversational language, not just data dumps
            - Suggest appropriate clothing or activities based on weather
            - Format responses in a user-friendly way"""
        )
        self.assigned_tools = mcp_tools_manager.get_tools_for_agent("weather")
        self.location_cache: Dict[str, str] = {}
        self.weather_cache: Dict[str, tuple] = {}
        self.cache_ttl = 900  # 15-minute cache
        
        log_structured("weather_agent_062_init",
                     autogen_version="0.6.2-official",
                     mcp_enabled=True)

    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Enhanced weather processing with follow-up generation"""
        try:
            if not self.is_weather_query(message):
                return await self.get_help_response()
            
            location = self.extract_location(message)
            is_forecast = "forecast" in message.lower()
            
            # Get weather data
            if is_forecast:
                result = await self.get_forecast_via_mcp(location)
            else:
                result = await self.get_current_via_mcp(location)
            
            # Generate contextual follow-up questions
            follow_up_questions = await self.generate_weather_follow_ups(
                message, result, location, context
            )
            
            # Combine weather info with follow-ups
            final_response = result
            if follow_up_questions:
                final_response += f"\n\n{follow_up_questions}"
            
            await self.store_weather_interaction(message, final_response, location)
            return final_response
            
        except Exception as e:
            log_structured("weather_fetch_failed", location=location, error=str(e))
            return f"I'm having trouble getting weather information for {location}. Please try again."

    async def get_current_via_mcp(self, location: str) -> str:
        """Get current weather using MCP tools"""
        try:
            # Get available MCP tools
            tools = await mcp_tools_manager.get_available_tools()
            
            # Find weather tool
            weather_tool = None
            for tool in tools:
                if "weather" in tool.name.lower() and "current" in tool.name.lower():
                    weather_tool = tool
                    break
            
            if weather_tool:
                result = await mcp_tools_manager.call_mcp_tool(
                    weather_tool.name,
                    {"location": location}
                )
                return self.format_weather_response(result, location)
            else:
                return self.demo_weather(location)
                
        except Exception as e:
            log_structured("current_weather_mcp_failed", location=location, error=str(e))
            return self.demo_weather(location)

    async def get_forecast_via_mcp(self, location: str) -> str:
        """Get weather forecast using MCP tools"""
        try:
            # Get available MCP tools
            tools = await mcp_tools_manager.get_available_tools()
            
            # Find forecast tool
            forecast_tool = None
            for tool in tools:
                if "weather" in tool.name.lower() and "forecast" in tool.name.lower():
                    forecast_tool = tool
                    break
            
            if forecast_tool:
                result = await mcp_tools_manager.call_mcp_tool(
                    forecast_tool.name,
                    {"location": location, "days": 1}
                )
                return self.format_forecast_response(result, location)
            else:
                return self.demo_forecast(location)
                
        except Exception as e:
            log_structured("forecast_mcp_failed", location=location, error=str(e))
            return self.demo_forecast(location)

    def format_weather_response(self, data: Dict[str, Any], location: str) -> str:
        """Format weather response with activity suggestions"""
        try:
            temp = data.get("temperature", 20)
            feels_like = data.get("feels_like", temp)
            condition = data.get("condition", "Clear")
            humidity = data.get("humidity", 50)
            wind_speed = data.get("wind_speed", 10)
            
            # Basic weather info
            response = f"""**Current weather for {location}:**
    🌡️ {temp:.1f}°C (feels like {feels_like:.1f}°C)
    🌤️ {condition}
    💧 Humidity: {humidity}%
    💨 Wind: {wind_speed:.1f} km/h"""
            
            # Add contextual recommendations
            recommendations = self.get_weather_recommendations(temp, condition, wind_speed)
            if recommendations:
                response += f"\n\n**Recommendations:**\n{recommendations}"
            
            return response
            
        except Exception:
            return self.demo_weather(location)

    def get_weather_recommendations(self, temp: float, condition: str, wind_speed: float) -> str:
        """Generate weather-based recommendations"""
        recommendations = []
        
        # Temperature-based recommendations
        if temp < 5:
            recommendations.append("🧥 Dress warmly - heavy coat recommended")
        elif temp < 15:
            recommendations.append("🧤 Light jacket or sweater recommended")
        elif temp > 25:
            recommendations.append("🌞 Perfect weather for outdoor activities")
        
        # Condition-based recommendations
        if "rain" in condition.lower():
            recommendations.append("☔ Don't forget your umbrella")
        elif "snow" in condition.lower():
            recommendations.append("❄️ Watch for slippery conditions")
        elif "clear" in condition.lower() or "sunny" in condition.lower():
            recommendations.append("🕶️ Great day for a walk or outdoor lunch")
        
        # Wind-based recommendations
        if wind_speed > 20:
            recommendations.append("🌬️ Quite windy - secure loose items")
        
        return "\n".join([f"• {rec}" for rec in recommendations])


    def format_forecast_response(self, data: Dict[str, Any], location: str) -> str:
        """Format forecast response from MCP tool data"""
        try:
            day_condition = data.get("day_condition", "Partly Cloudy")
            night_condition = data.get("night_condition", "Clear")
            min_temp = data.get("min_temperature", 15)
            max_temp = data.get("max_temperature", 25)
            
            return f"""Today's forecast for {location}:
Day: {day_condition}
Night: {night_condition}
Range: {min_temp:.1f}°C - {max_temp:.1f}°C"""
            
        except Exception:
            return self.demo_forecast(location)

    def extract_location(self, message: str) -> str:
        """Extract location from message"""
        tokens = message.lower().split()
        for idx, token in enumerate(tokens):
            if token in ["in", "for", "at"] and idx + 1 < len(tokens):
                return tokens[idx + 1].title()
        return "Brussels"

    def is_weather_query(self, message: str) -> bool:
        """Check if message is weather-related"""
        keywords = ["weather", "temperature", "forecast", "rain", "snow", "wind", "sunny"]
        return any(keyword in message.lower() for keyword in keywords)

    async def get_help_response(self) -> str:
        """Generate help response using AutoGen 0.6.2"""
        return """I can help with weather information! Try questions like:
- What's the weather in Brussels?
- Weather forecast for tomorrow
- Temperature in Paris

I provide accurate weather data and helpful recommendations."""

    def demo_weather(self, location: str) -> str:
        """Demo weather response"""
        return f"""Current weather for {location}:
19.0°C (feels like 21.0°C)
Partly cloudy
Humidity: 60%
Wind: 13 km/h
[Demo mode - MCP tools unavailable]"""

    def demo_forecast(self, location: str) -> str:
        """Demo forecast response"""
        return f"""Today's forecast for {location}:
Day: Partly Cloudy
Night: Clear
Range: 15°C - 21°C
[Demo mode - MCP tools unavailable]"""

    async def store_weather_interaction(self, user_message: str, response: str, location: str):
        """Store weather interaction in memory"""
        try:
            await self.vector_memory.add(
                content=f"Weather Query: {user_message}\nResponse: {response[:200]}",
                metadata={
                    "type": "weather_interaction",
                    "agent_id": self.agent_id,
                    "session_id": self.current_session_id,
                    "location": location,
                    "autogen_version": "0.6.2-official"
                }
            )
        except Exception as e:
            log_structured("weather_memory_store_fail", error=str(e))



    async def generate_weather_follow_ups(self, user_message: str, weather_response: str, location: str, context: Dict[str, Any] = None) -> str:
        """Generate contextual follow-up questions for weather queries"""
        try:
            # Analyze what information might be missing or helpful
            follow_up_context = {
                "user_message": user_message,
                "weather_response": weather_response,
                "location": location,
                "conversation_history": context.get("conversation_history", []) if context else [],
                "time_context": context.get("time_context", {}) if context else {},
                "intent": "weather_query"
            }
            
            # Generate follow-up questions
            follow_ups = await follow_up_generator.generate_follow_ups(
                intent="weather_query",
                context=follow_up_context,
                agent_type="weather"
            )
            
            if follow_ups:
                return f"**Would you also like to know:**\n" + "\n".join([f"• {q}" for q in follow_ups])
            
            return ""
            
        except Exception as e:
            log_structured("weather_follow_up_generation_failed", error=str(e))
            return ""
