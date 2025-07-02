"""
Direct Weather Agent - AccuWeather API Integration
Fast, reliable weather data with behavior learning integration
"""
import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from autogen_core import MessageContext
import requests

from src.agents.core.base_agent import AutoGenBaseAgent
from src.agents.core.logging_config import log_structured


class ReliableWeatherAgent(AutoGenBaseAgent):
    """Weather agent that fetches weather data directly from AccuWeather API"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Weather information specialist with AccuWeather API integration",
            agent_type="weather"
        )
        
        # AccuWeather API configuration
        self.accuweather_api_key = os.getenv("ACCUWEATHER_API_KEY")
        self.demo_mode = not self.accuweather_api_key
        self.location_search_url = "http://dataservice.accuweather.com/locations/v1/cities/search"
        self.current_weather_url = "http://dataservice.accuweather.com/currentconditions/v1"
        self.forecast_url = "http://dataservice.accuweather.com/forecasts/v1/daily/1day"
        
        # Cache for location keys
        self.location_cache = {}
        
        self._initialized = True
        
        log_structured("weather_agent_accuweather_init", 
                      api_available=bool(self.accuweather_api_key),
                      mode="real" if not self.demo_mode else "demo")

    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process weather messages with direct AccuWeather API calls"""
        try:
            log_structured("weather_processing_start", 
                          message=message[:100], 
                          session_id=self.current_session_id,
                          api_mode="accuweather_direct")
            
            if not self._is_weather_query(message):
                return self._provide_weather_help()
            
            # Try behavior learning first for context
            response = await self._process_with_memory_context(message)
            if response:
                await self._store_interaction(message, response)
                return response
            
            # Get fresh weather data
            response = await self._get_fresh_weather(message)
            await self._store_interaction(message, response)
            return response
            
        except Exception as e:
            log_structured("weather_processing_error", error=str(e))
            return f"⚠️ Weather service temporarily unavailable: {str(e)}"

    def _is_weather_query(self, message: str) -> bool:
        """Detect weather queries"""
        weather_keywords = [
            "weather", "temperature", "forecast", "rain", "sunny", "cloudy", 
            "hot", "cold", "humid", "wind", "storm", "snow", "climate",
            "degrees", "celsius", "fahrenheit", "precipitation"
        ]
        return any(keyword in message.lower() for keyword in weather_keywords)

    async def _process_with_memory_context(self, message: str) -> Optional[str]:
        """Use behavior learning for enhanced context"""
        try:
            from src.learning.behavior.behavior_engine import search_semantic_memory
            
            memory_results = search_semantic_memory(
                message, n_results=3, session_id=self.current_session_id
            )
            
            if memory_results and memory_results.get("documents"):
                return await self._enhance_with_memory_context(message, memory_results)
            
            return None
            
        except ImportError:
            log_structured("behavior_learning_unavailable")
            return None
        except Exception as e:
            log_structured("behavior_learning_error", error=str(e))
            return None

    async def _enhance_with_memory_context(self, message: str, memory_results: Dict) -> str:
        """Combine current weather with memory context"""
        try:
            location = self._extract_location(message)
            current_weather = await self._fetch_current_weather(location)
            
            response_parts = [f"🌤️ **Current weather for {location}:**"]
            response_parts.append(current_weather)
            
            # Add memory context
            previous_locations = self._extract_previous_locations(memory_results)
            if previous_locations and len(previous_locations) > 1:
                unique_locations = list(set(previous_locations[:3]))
                if location.lower() not in [loc.lower() for loc in unique_locations]:
                    response_parts.append(f"\n💭 *I remember you've also asked about: {', '.join(unique_locations)}*")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            log_structured("memory_context_error", error=str(e))
            return await self._get_fresh_weather(message)

    async def _get_fresh_weather(self, message: str) -> str:
        """Get fresh weather data"""
        location = self._extract_location(message)
        
        if "forecast" in message.lower():
            return await self._get_forecast(location)
        else:
            return await self._fetch_current_weather(location)

    async def _get_location_key(self, location: str) -> Optional[str]:
        """Get AccuWeather location key for a city"""
        try:
            if location in self.location_cache:
                return self.location_cache[location]
                
            if self.demo_mode:
                return None
            
            params = {
                "apikey": self.accuweather_api_key,
                "q": location
            }
            
            log_structured("accuweather_location_search", location=location)
            
            response = requests.get(self.location_search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                location_key = data[0]["Key"]
                self.location_cache[location] = location_key
                
                log_structured("accuweather_location_found", 
                             location=location, 
                             location_key=location_key,
                             full_name=data[0].get("LocalizedName", location))
                
                return location_key
            
            return None
            
        except Exception as e:
            log_structured("accuweather_location_error", location=location, error=str(e))
            return None

    async def _fetch_current_weather(self, location: str) -> str:
        """Fetch current weather from AccuWeather API"""
        try:
            if self.demo_mode:
                return self._get_demo_weather(location)
            
            # Get location key first
            location_key = await self._get_location_key(location)
            if not location_key:
                log_structured("accuweather_no_location_key", location=location)
                return self._get_demo_weather(location)
            
            # Get current conditions
            params = {
                "apikey": self.accuweather_api_key,
                "details": "true"
            }
            
            url = f"{self.current_weather_url}/{location_key}"
            
            log_structured("accuweather_api_call", location=location, endpoint="current")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                weather_data = data[0]
                
                weather_info = {
                    "location": location,
                    "temperature": f"{weather_data['Temperature']['Metric']['Value']:.1f}°C",
                    "feels_like": f"{weather_data.get('RealFeelTemperature', {}).get('Metric', {}).get('Value', 'N/A'):.1f}°C" if weather_data.get('RealFeelTemperature', {}).get('Metric', {}).get('Value') else "N/A",
                    "conditions": weather_data['WeatherText'],
                    "humidity": f"{weather_data.get('RelativeHumidity', 'N/A')}%" if weather_data.get('RelativeHumidity') else "N/A",
                    "pressure": f"{weather_data.get('Pressure', {}).get('Metric', {}).get('Value', 'N/A')} mb" if weather_data.get('Pressure', {}).get('Metric', {}).get('Value') else "N/A",
                    "wind_speed": f"{weather_data.get('Wind', {}).get('Speed', {}).get('Metric', {}).get('Value', 0):.1f} km/h" if weather_data.get('Wind', {}).get('Speed', {}).get('Metric', {}).get('Value') else "N/A",
                    "visibility": f"{weather_data.get('Visibility', {}).get('Metric', {}).get('Value', 'N/A')} km" if weather_data.get('Visibility', {}).get('Metric', {}).get('Value') else "N/A",
                    "demo": False
                }
                
                log_structured("accuweather_api_success", 
                             location=location, 
                             temperature=weather_info["temperature"])
                
                return self._format_weather_response(weather_info)
            
            return self._get_demo_weather(location)
            
        except requests.exceptions.RequestException as e:
            log_structured("accuweather_api_error", location=location, error=str(e))
            return self._get_demo_weather(location)
        except Exception as e:
            log_structured("accuweather_fetch_error", location=location, error=str(e))
            return self._get_demo_weather(location)

    async def _get_forecast(self, location: str) -> str:
        """Get weather forecast from AccuWeather"""
        try:
            if self.demo_mode:
                return self._get_demo_forecast(location)
            
            # Get location key first
            location_key = await self._get_location_key(location)
            if not location_key:
                return self._get_demo_forecast(location)
            
            # Get forecast
            params = {
                "apikey": self.accuweather_api_key,
                "details": "true",
                "metric": "true"
            }
            
            url = f"{self.forecast_url}/{location_key}"
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and "DailyForecasts" in data and len(data["DailyForecasts"]) > 0:
                forecast = data["DailyForecasts"][0]
                
                forecast_parts = [f"🌤️ **Today's forecast for {location}:**"]
                
                # Day forecast
                if "Day" in forecast:
                    day_forecast = forecast["Day"]
                    forecast_parts.append(f"🌅 **Day**: {day_forecast.get('IconPhrase', 'N/A')}")
                
                # Night forecast  
                if "Night" in forecast:
                    night_forecast = forecast["Night"]
                    forecast_parts.append(f"🌙 **Night**: {night_forecast.get('IconPhrase', 'N/A')}")
                
                # Temperature range
                if "Temperature" in forecast:
                    temp = forecast["Temperature"]
                    min_temp = temp.get("Minimum", {}).get("Value", "N/A")
                    max_temp = temp.get("Maximum", {}).get("Value", "N/A")
                    forecast_parts.append(f"🌡️ **Range**: {min_temp}°C - {max_temp}°C")
                
                return "\n".join(forecast_parts)
            
            return self._get_demo_forecast(location)
            
        except Exception as e:
            log_structured("accuweather_forecast_error", location=location, error=str(e))
            return self._get_demo_forecast(location)

    def _format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        """Format weather response"""
        try:
            parts = [
                f"📍 **{weather_data['location']}**",
                f"🌡️ {weather_data['temperature']}" + (f" (feels like {weather_data['feels_like']})" if weather_data['feels_like'] != "N/A" else ""),
                f"☁️ {weather_data['conditions']}"
            ]
            
            if weather_data['humidity'] != "N/A":
                parts.append(f"💧 Humidity: {weather_data['humidity']}")
            
            if weather_data['wind_speed'] != "N/A":
                parts.append(f"🌬️ Wind: {weather_data['wind_speed']}")
            
            if weather_data['visibility'] != "N/A":
                parts.append(f"👁️ Visibility: {weather_data['visibility']}")
            
            if weather_data.get("demo", False):
                parts.append("*(Demo mode - API unavailable)*")
            
            return "\n".join(parts)
            
        except Exception:
            return str(weather_data)

    def _extract_location(self, message: str) -> str:
        """Extract location from message"""
        words = message.lower().split()
        
        # Look for location indicators
        location_indicators = ["in", "for", "at", "from", "near", "around"]
        for i, word in enumerate(words):
            if word in location_indicators and i + 1 < len(words):
                return words[i + 1].strip("?.,!").title()
        
        # Check for known cities
        cities = {
            "brussels": "Brussels", "etterbeek": "Etterbeek", "paris": "Paris", 
            "london": "London", "amsterdam": "Amsterdam", "berlin": "Berlin",
            "antwerp": "Antwerp", "ghent": "Ghent", "bruges": "Bruges",
            "new york": "New York", "tokyo": "Tokyo", "sydney": "Sydney"
        }
        
        for word in words:
            clean_word = word.strip("?.,!")
            if clean_word in cities:
                return cities[clean_word]
        
        return "Brussels"  # Default

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

    def _get_demo_weather(self, location: str) -> str:
        """Get demo weather data"""
        demo_data = {
            "brussels": {
                "location": "Brussels, BE",
                "temperature": "18.5°C",
                "feels_like": "16.2°C",
                "conditions": "Partly Cloudy",
                "humidity": "65%",
                "wind_speed": "12.5 km/h",
                "visibility": "10.0 km",
                "demo": True
            },
            "etterbeek": {
                "location": "Etterbeek, BE", 
                "temperature": "18.2°C",
                "feels_like": "16.0°C",
                "conditions": "Mild",
                "humidity": "63%",
                "wind_speed": "10.8 km/h",
                "visibility": "10.0 km",
                "demo": True
            },
            "paris": {
                "location": "Paris, FR",
                "temperature": "20.1°C",
                "feels_like": "19.5°C",
                "conditions": "Sunny",
                "humidity": "55%",
                "wind_speed": "15.2 km/h",
                "visibility": "15.0 km",
                "demo": True
            }
        }
        
        weather_data = demo_data.get(location.lower(), {
            "location": f"{location}",
            "temperature": "19.0°C",
            "feels_like": "17.5°C",
            "conditions": "Mild",
            "humidity": "60%",
            "wind_speed": "13.5 km/h",
            "visibility": "10.0 km",
            "demo": True
        })
        
        return self._format_weather_response(weather_data)

    def _get_demo_forecast(self, location: str) -> str:
        """Get demo forecast"""
        return f"""🌤️ **Today's forecast for {location}:**
🌅 **Day**: Partly Cloudy
🌙 **Night**: Clear
🌡️ **Range**: 15.5°C - 21.2°C
*(Demo mode - API unavailable)*"""

    def _provide_weather_help(self) -> str:
        """Provide weather help"""
        api_status = "✅ AccuWeather API" if not self.demo_mode else "⚠️ Demo mode"
        
        return f"""🌤️ **I can help with weather information!** ({api_status})

**Try asking:**
• *"What's the weather in Brussels?"*
• *"Weather forecast for tomorrow"*
• *"Temperature in Etterbeek"*
• *"Is it raining in Paris?"*

I use AccuWeather API for accurate, real-time weather data."""

    async def _store_interaction(self, message: str, response: str):
        """Store interaction in database and behavior learning"""
        try:
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)
            
            interaction_id = self.database.store_agent_interaction(
                agent_id=self.agent_id,
                interaction_type="weather_query",
                data={
                    "user_message": message,
                    "agent_response": response,
                    "session_id": self.current_session_id,
                    "api_mode": "accuweather_direct",
                    "demo_mode": self.demo_mode,
                    "location": self._extract_location(message)
                }
            )
            
            # Add to semantic memory for behavior learning
            try:
                from src.learning.behavior.behavior_engine import add_to_semantic_memory
                await add_to_semantic_memory(
                    content=f"Weather Query: {message} | Response: {response[:200]}",
                    metadata={
                        "type": "weather_interaction",
                        "agent_id": self.agent_id,
                        "session_id": self.current_session_id,
                        "interaction_id": interaction_id,
                        "domain": "weather",
                        "location": self._extract_location(message),
                        "api_mode": "accuweather_direct"
                    }
                )
            except ImportError:
                log_structured("behavior_learning_unavailable")
            except Exception as memory_error:
                log_structured("behavior_learning_storage_error", error=str(memory_error))
            
            log_structured("weather_interaction_stored", 
                         interaction_id=interaction_id,
                         location=self._extract_location(message))
            
        except Exception as e:
            log_structured("weather_interaction_storage_error", error=str(e))
