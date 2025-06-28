# api/mcp_weather.py - OFFICIAL MCP PROTOCOL
#!/usr/bin/env python3
import asyncio
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from dotenv import load_dotenv
import logging

# OFFICIAL MCP SDK
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# OFFICIAL: Initialize MCP server
mcp = FastMCP("weather")

# Cache configuration
CACHE_DIR = Path("/app/.cache/weather")
LOCATION_CACHE_FILE = CACHE_DIR / "location_cache.json"

# AccuWeather API constants
ACCUWEATHER_BASE = "http://dataservice.accuweather.com"

def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cached_location_key(location: str) -> Optional[str]:
    """Get cached location key."""
    if not LOCATION_CACHE_FILE.exists():
        return None
    
    try:
        with open(LOCATION_CACHE_FILE, "r") as f:
            cache = json.load(f)
            return cache.get(location.lower())
    except:
        return None

def cache_location_key(location: str, location_key: str):
    """Cache location key."""
    ensure_cache_dir()
    
    try:
        cache = {}
        if LOCATION_CACHE_FILE.exists():
            with open(LOCATION_CACHE_FILE, "r") as f:
                cache = json.load(f)
        
        cache[location.lower()] = location_key
        
        with open(LOCATION_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to cache location: {e}")

@mcp.tool()
async def get_weather(location: str = "Brussels") -> Dict[str, Any]:
    """Get current weather conditions for a location.
    
    Args:
        location: Location name (city, state/country)
    
    Returns:
        Weather information including temperature, conditions, humidity
    """
    logger.info(f"Weather request for: {location}")
    
    api_key = os.getenv("ACCUWEATHER_API_KEY")
    if not api_key:
        logger.warning("No AccuWeather API key configured")
        return {
            "location": location,
            "temperature": "20°C",
            "conditions": "Demo mode - Configure ACCUWEATHER_API_KEY for real data",
            "humidity": "65%",
            "demo": True,
            "mcp_server": "weather-sse"
        }
    
    location_key = get_cached_location_key(location)
    
    try:
        async with httpx.AsyncClient() as client:
            # Get location key if not cached
            if not location_key:
                logger.info(f"Looking up location key for: {location}")
                search_url = f"{ACCUWEATHER_BASE}/locations/v1/cities/search"
                params = {"apikey": api_key, "q": location}
                
                response = await client.get(search_url, params=params)
                if response.status_code != 200:
                    return {
                        "error": f"Location search failed: {response.status_code}",
                        "location": location,
                        "temperature": "N/A",
                        "conditions": "API Error"
                    }
                
                locations = response.json()
                if not locations:
                    return {
                        "error": "Location not found",
                        "location": location,
                        "temperature": "N/A",
                        "conditions": "Location not found"
                    }
                
                location_data = locations[0]
                location_key = location_data["Key"]
                cache_location_key(location, location_key)
                logger.info(f"Cached location key: {location_key}")
            
            # Get current conditions
            conditions_url = f"{ACCUWEATHER_BASE}/currentconditions/v1/{location_key}"
            params = {"apikey": api_key, "details": "true"}
            
            response = await client.get(conditions_url, params=params)
            if response.status_code != 200:
                return {
                    "error": f"Weather API error: {response.status_code}",
                    "location": location,
                    "temperature": "N/A",
                    "conditions": "API Error"
                }
            
            conditions = response.json()
            if not conditions:
                return {
                    "error": "No weather data available",
                    "location": location,
                    "temperature": "N/A",
                    "conditions": "No data"
                }
            
            current = conditions[0]
            
            result = {
                "location": location,
                "temperature": f"{current['Temperature']['Metric']['Value']}°C",
                "conditions": current.get("WeatherText", "Unknown"),
                "humidity": f"{current.get('RelativeHumidity', 'N/A')}%",
                "observation_time": current.get("LocalObservationDateTime", "Unknown"),
                "mcp_server": "weather-sse",
                "api_used": "AccuWeather"
            }
            
            logger.info(f"Weather response: {result['location']} - {result['temperature']}")
            return result
            
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return {
            "error": str(e),
            "location": location,
            "temperature": "N/A",
            "conditions": "Service Error",
            "mcp_server": "weather-sse"
        }

@mcp.tool()
async def get_forecast(location: str = "Brussels", days: int = 3) -> Dict[str, Any]:
    """Get weather forecast for a location.
    
    Args:
        location: Location name
        days: Number of days (1-5)
    
    Returns:
        Weather forecast information
    """
    api_key = os.getenv("ACCUWEATHER_API_KEY")
    if not api_key:
        return {
            "location": location,
            "forecast": f"Demo {days}-day forecast for {location}",
            "days": days,
            "demo": True,
            "mcp_server": "weather-sse"
        }
    
    # Implementation would go here for real forecast
    return {
        "location": location,
        "forecast": f"Forecast available with API key for {days} days",
        "days": days,
        "mcp_server": "weather-sse"
    }



if __name__ == "__main__":
    logger.info("Starting Official MCP Weather Server (SSE Transport)")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"API key configured: {bool(os.getenv('ACCUWEATHER_API_KEY'))}")
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=8182,
        log_level="INFO"
    )
    


