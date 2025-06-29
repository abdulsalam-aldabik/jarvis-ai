#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, Any

from fastmcp import FastMCP

# Configure logging for faster startup
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

# Initialize FastMCP with minimal configuration
mcp = FastMCP("weather")

# Pre-computed demo data for instant response
DEMO_WEATHER_DATA = {
    "Brussels": {"temp": "18°C", "conditions": "Partly cloudy", "humidity": "72%"},
    "Paris": {"temp": "20°C", "conditions": "Sunny", "humidity": "65%"},
    "London": {"temp": "15°C", "conditions": "Cloudy", "humidity": "80%"},
    "Amsterdam": {"temp": "17°C", "conditions": "Light rain", "humidity": "85%"}
}

@mcp.tool()
def get_weather(location: str = "Brussels") -> Dict[str, Any]:
    """Get current weather conditions instantly."""
    # OPTIMIZATION: Return immediately without any delays
    try:
        # Get demo data or default
        weather_data = DEMO_WEATHER_DATA.get(location, DEMO_WEATHER_DATA["Brussels"])
        
        # Return minimal, structured response immediately
        return {
            "location": location,
            "temperature": weather_data["temp"],
            "conditions": weather_data["conditions"],
            "humidity": weather_data["humidity"],
            "status": "success"
        }
    except Exception:
        # Even errors return immediately
        return {"location": location, "error": "unavailable", "status": "error"}

@mcp.tool()
def get_forecast(location: str = "Brussels", days: int = 3) -> Dict[str, Any]:
    """Get weather forecast instantly."""
    try:
        # Limit days and return immediately
        days = min(max(days, 1), 3)  # 1-3 days only
        
        return {
            "location": location,
            "days": days,
            "forecast": [
                {"day": 1, "temp": "19°C", "conditions": "Sunny"},
                {"day": 2, "temp": "21°C", "conditions": "Cloudy"},
                {"day": 3, "temp": "18°C", "conditions": "Rain"}
            ][:days],
            "status": "success"
        }
    except Exception:
        return {"location": location, "error": "unavailable", "status": "error"}

if __name__ == "__main__":
    logger.warning("Starting Ultra-Fast Weather Server")
    
    # OPTIMIZATION: Minimal server configuration for speed
    try:
        mcp.run(
            transport="sse",
            host="0.0.0.0",
            port=8182,
            log_level="WARNING"  # Reduce logging overhead
        )
    except Exception as e:
        logger.error(f"Server failed: {e}")
        raise
