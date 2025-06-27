"""
Weather agent compatible with corrected AutoGen 2.x patterns
"""
import aiohttp
import asyncio
from autogen_core import MessageContext
from agents.core.base_agent import AutoGenBaseAgent
from config.settings import settings
from agents.core.database import db_manager
from agents.core.logging_config import log_structured
from typing import Dict, Any, Optional

class ReliableWeatherAgent(AutoGenBaseAgent):
    """Reliable AutoGen weather agent with proper session context"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Weather information and forecasting with session awareness",
            agent_type="weather"
        )

    def log_reasoning(self, step: str, reasoning: str, data: Dict[str, Any] = None):
        """Log reasoning with structured logging"""
        try:
            log_structured("weather_reasoning", 
                         step=step, 
                         reasoning=reasoning, 
                         data=data or {}, 
                         agent_id=self.agent_id,
                         session_id=self.current_session_id)
        except Exception as e:
            log_structured("weather_reasoning_failed", error=str(e), agent_id=self.agent_id)

    def check_configuration(self) -> Dict[str, Any]:
        """Check weather API configuration"""
        config_status = {
            "api_key_configured": bool(settings.api.accuweather_key),
            "api_key_length": len(settings.api.accuweather_key) if settings.api.accuweather_key else 0,
            "api_url_base": "https://dataservice.accuweather.com",
            "timeout_seconds": 15
        }
        
        self.log_reasoning("config_check", "Weather agent configuration check", config_status)
        return config_status

    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process weather requests with session context"""
        try:
            location = self.extract_location(message)
            config = self.check_configuration()
            
            self.log_reasoning("weather_request", 
                             f"Processing weather request for {location}", 
                             {"message": message, "config": config})
            
            weather_data = await self.get_weather_data(location)
            response = self.format_weather_response(weather_data, location)
            
            # Store interaction with session context
            await self._store_interaction(message, response)
            return response
            
        except Exception as e:
            self.log_reasoning("weather_error", f"Weather processing failed: {str(e)}", 
                             {"message": message, "session_id": self.current_session_id})
            return f"I'm having trouble getting weather information: {str(e)[:50]}"

    def extract_location(self, message: str) -> str:
        """Extract location from message"""
        import re
        
        patterns = [
            r"(?:in|for|at)\s+([A-Za-z\s]{2,30})(?:[?.!,]|$)",
            r"weather\s+([A-Za-z\s]{2,30})(?:[?.!,]|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) > 1:
                    return location
        
        return "Brussels"  # Default location

    async def get_weather_data(self, location: str) -> dict:
        """Get weather data from AccuWeather API"""
        if not settings.api.accuweather_key:
            self.log_reasoning("api_key_missing", "AccuWeather API key not configured", {"location": location})
            return {"success": False, "message": "API key not configured"}
            
        if len(settings.api.accuweather_key) < 10:
            self.log_reasoning("api_key_invalid", "AccuWeather API key appears invalid", 
                             {"key_length": len(settings.api.accuweather_key)})
            return {"success": False, "message": "API key appears invalid"}

        try:
            async with aiohttp.ClientSession() as session:
                # Step 1: Search for location key
                location_url = "https://dataservice.accuweather.com/locations/v1/cities/search"
                location_params = {
                    "apikey": settings.api.accuweather_key,
                    "q": location,
                    "details": "false"
                }
                
                self.log_reasoning("api_call", f"Calling AccuWeather API for {location}", {"url": location_url})
                
                async with session.get(location_url, params=location_params, timeout=10) as response:
                    if response.status == 401:
                        self.log_reasoning("api_auth_failed", "AccuWeather API authentication failed", 
                                         {"status": response.status})
                        return {"success": False, "message": "API key invalid or expired"}
                    
                    if response.status == 403:
                        self.log_reasoning("api_quota_exceeded", "AccuWeather API quota exceeded", 
                                         {"status": response.status})
                        return {"success": False, "message": "API quota exceeded"}
                    
                    if response.status != 200:
                        self.log_reasoning("api_error", f"AccuWeather API error {response.status}", 
                                         {"status": response.status})
                        return {"success": False, "message": f"API error {response.status}"}
                    
                    locations = await response.json()
                    
                    if not locations:
                        self.log_reasoning("location_not_found", f"No location found for {location}")
                        return {"success": False, "message": f"Location '{location}' not found"}
                    
                    location_key = locations[0]["Key"]
                    location_name = locations[0]["LocalizedName"]
                    country_name = locations[0]["Country"]["LocalizedName"]

                # Step 2: Get current conditions
                conditions_url = f"https://dataservice.accuweather.com/currentconditions/v1/{location_key}"
                conditions_params = {
                    "apikey": settings.api.accuweather_key,
                    "details": "true"
                }
                
                async with session.get(conditions_url, params=conditions_params, timeout=10) as response:
                    if response.status != 200:
                        self.log_reasoning("conditions_api_error", 
                                         f"Current conditions API error {response.status}", 
                                         {"status": response.status})
                        return {"success": False, "message": f"Weather API error {response.status}"}
                    
                    conditions = await response.json()
                    
                    if not conditions:
                        self.log_reasoning("conditions_not_found", "No current conditions returned")
                        return {"success": False, "message": "No weather data found"}
                    
                    current = conditions[0]
                    
                    result = {
                        "success": True,
                        "location": location_name,
                        "country": country_name,
                        "temperature": current["Temperature"]["Metric"]["Value"],
                        "temperature_unit": current["Temperature"]["Metric"]["Unit"],
                        "conditions": current["WeatherText"],
                        "humidity": current.get("RelativeHumidity"),
                        "observation_time": current["LocalObservationDateTime"]
                    }
                    
                    self.log_reasoning("weather_data_success", "Successfully retrieved weather data", result)
                    return result
                    
        except asyncio.TimeoutError:
            self.log_reasoning("api_timeout", f"AccuWeather API timeout for {location}", {"timeout": 10})
            return {"success": False, "message": "API timeout"}
        except Exception as e:
            self.log_reasoning("api_exception", f"AccuWeather API exception: {str(e)}", {"location": location})
            return {"success": False, "message": f"API error: {str(e)[:50]}"}

    def format_weather_response(self, weather_data: dict, location: str) -> str:
        """Format weather response for user"""
        if weather_data.get("success"):
            temp = weather_data["temperature"]
            unit = weather_data["temperature_unit"]
            conditions = weather_data["conditions"]
            location_name = weather_data["location"]
            country = weather_data["country"]
            
            response = f"Weather in {location_name}, {country}: {temp}°{unit}, {conditions}"
            
            if weather_data.get("humidity") is not None:
                response += f". Humidity: {weather_data['humidity']}%"
            
            self.log_reasoning("response_formatted", 
                             f"Formatted weather response for {location_name}", 
                             {"response_length": len(response)})
            return response
        else:
            error_message = weather_data.get("message", "Unknown error")
            return f"Weather service error for {location}: {error_message}"
