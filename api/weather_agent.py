#!/usr/bin/env python3
"""
MCP Weather Agent Server
Provides weather information via MCP protocol
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

try:
    from fastmcp import FastMCP
    import aiohttp
    import logging
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required dependency: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherMCPServer:
    """MCP Server for weather information"""
    
    def __init__(self):
        self.api_key = os.getenv("ACCUWEATHER_API_KEY")
        logger.info(f"API Key configured: {bool(self.api_key)}")
        if self.api_key:
            logger.info(f"API Key length: {len(self.api_key)}")
    
    async def get_weather(self, location: str = "Brussels") -> Dict[str, Any]:
        """Get current weather for a location"""
        
        if not self.api_key:
            logger.warning("AccuWeather API key not configured")
            return {
                "error": "AccuWeather API key not configured",
                "location": location,
                "temperature": "15°C",
                "conditions": "Demo data - API key required for real weather",
                "status": "demo_mode"
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get location key
                location_url = "http://dataservice.accuweather.com/locations/v1/cities/search"
                location_params = {
                    "apikey": self.api_key,
                    "q": location,
                    "details": "false"
                }
                
                logger.info(f"Requesting weather for {location}")
                
                async with session.get(location_url, params=location_params, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"Location API error: {response.status}")
                        raise Exception(f"Location API error: {response.status}")
                    
                    locations = await response.json()
                    if not locations:
                        return {
                            "error": f"Location '{location}' not found",
                            "location": location,
                            "temperature": "N/A",
                            "conditions": "Location not found"
                        }
                    
                    location_key = locations[0]["Key"]
                    location_name = locations[0]["LocalizedName"]
                    country = locations[0]["Country"]["LocalizedName"]
                
                # Get current conditions
                current_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}"
                current_params = {"apikey": self.api_key, "details": "true"}
                
                async with session.get(current_url, params=current_params, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"Weather API error: {response.status}")
                        raise Exception(f"Weather API error: {response.status}")
                    
                    conditions = await response.json()
                    if not conditions:
                        raise Exception("No weather data returned")
                    
                    current = conditions[0]
                    
                    result = {
                        "success": True,
                        "location": location_name,
                        "country": country,
                        "temperature": f"{current['Temperature']['Metric']['Value']}°C",
                        "conditions": current["WeatherText"],
                        "humidity": current.get("RelativeHumidity"),
                        "observation_time": current["LocalObservationDateTime"]
                    }
                    
                    logger.info(f"Successfully retrieved weather for {location_name}: {result['temperature']}")
                    return result
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting weather for {location}")
            return {
                "error": "Weather service timeout",
                "location": location,
                "temperature": "N/A",
                "conditions": "Service timeout"
            }
        except Exception as e:
            logger.error(f"Weather API error for {location}: {e}")
            return {
                "error": str(e),
                "location": location,
                "temperature": "N/A",
                "conditions": "Service error"
            }

# Initialize MCP server
mcp = FastMCP("Weather Agent")
weather_server = WeatherMCPServer()

@mcp.tool()
async def get_weather(location: str = "Brussels") -> Dict[str, Any]:
    """
    Get current weather information for a location
    
    Args:
        location: The city name to get weather for (default: Brussels)
    
    Returns:
        Weather information including temperature, conditions, and location details
    """
    result = await weather_server.get_weather(location)
    return result

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Check the health of the weather service
    
    Returns:
        Health status and configuration information
    """
    return {
        "status": "healthy",
        "service": "Weather MCP Server",
        "api_key_configured": bool(weather_server.api_key),
        "capabilities": ["get_weather"]
    }

if __name__ == "__main__":
    logger.info("Starting Weather MCP Server...")
    mcp.run()
