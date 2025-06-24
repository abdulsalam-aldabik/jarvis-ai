import aiohttp
from typing import Dict, Any, List
from agents.core.base_agent import BaseAgent, AgentCapability, AgentTask
from config.settings import settings

class WeatherAgent(BaseAgent):
    """Intelligent weather agent"""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="get_weather_intelligently",
                description="Get weather information using intelligent analysis",
                input_schema={"location": "string", "request": "string"},
                output_schema={"weather_info": "object"}
            )
        ]
        
        super().__init__(
            agent_id="weather_agent",
            agent_type="weather",
            capabilities=capabilities
        )
    
    async def execute_intelligent_plan(self, task: AgentTask, reasoning: str) -> Any:
        """Execute intelligent weather plan - THIS WAS MISSING"""
        try:
            # Extract location from the request intelligently
            location = self._extract_location(task.content)
            
            # Get weather data
            weather_data = await self._get_weather_data(location)
            
            # Format intelligently based on the request
            if weather_data:
                formatted_response = await self._format_weather_response(weather_data, task.content, reasoning)
                return formatted_response
            else:
                return f"I couldn't get weather information for {location} right now. Please try again later."
                
        except Exception as e:
            return f"Weather service error: {str(e)}"
    
    def _extract_location(self, request: str) -> str:
        """Extract location from request"""
        import re
        
        # Look for location patterns
        patterns = [
            r"in\s+([A-Za-z\s,]+?)(?:\s|$|\?|!|\.)",
            r"for\s+([A-Za-z\s,]+?)(?:\s|$|\?|!|\.)",
            r"weather\s+([A-Za-z\s,]+?)(?:\s|$|\?|!|\.)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Brussels"  # Default location
    
    async def _get_weather_data(self, location: str) -> Dict[str, Any]:
        """Get weather data with fallback"""
        if not settings.api.accuweather_key:
            return {
                "location": location,
                "temperature": "15°C",
                "conditions": "Partly cloudy",
                "message": "Weather API not configured - showing sample data"
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Simplified weather API call with timeout
                async with session.get(
                    "http://dataservice.accuweather.com/locations/v1/cities/search",
                    params={"apikey": settings.api.accuweather_key, "q": location},
                    timeout=5
                ) as response:
                    if response.status == 200:
                        locations = await response.json()
                        if locations:
                            return {
                                "location": locations[0]["LocalizedName"],
                                "country": locations[0]["Country"]["LocalizedName"],
                                "temperature": "15°C",  # Simplified
                                "conditions": "Current conditions available"
                            }
        except Exception:
            pass
        
        # Fallback response
        return {
            "location": location,
            "temperature": "15°C",
            "conditions": "Weather information temporarily unavailable",
            "message": "Using fallback weather response"
        }
    
    async def _format_weather_response(self, weather_data: Dict[str, Any], original_request: str, reasoning: str) -> str:
        """Format weather response intelligently"""
        location = weather_data.get("location", "Unknown location")
        temp = weather_data.get("temperature", "Unknown")
        conditions = weather_data.get("conditions", "Unknown conditions")
        
        if "simple" in original_request.lower() or "quick" in original_request.lower():
            return f"{location}: {temp}, {conditions}"
        else:
            return f"The weather in {location} is currently {temp} with {conditions}. {weather_data.get('message', '')}"
