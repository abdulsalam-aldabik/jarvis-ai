#!/usr/bin/env python3

import os
import logging
from fastmcp import FastMCP
from dotenv import load_dotenv
import aiohttp
import asyncio

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WeatherMCPServer:
    def __init__(self):
        self.api_key = os.getenv("ACCUWEATHER_API_KEY")
        logger.info(f"API Key configured: {bool(self.api_key)}")

    async def get_weather(self, location: str = "Brussels"):
        if not self.api_key:
            return {"error": "API key not configured", "location": location, "temperature": "N/A", "conditions": "Demo mode"}
        try:
            async with aiohttp.ClientSession() as session:
                location_url = "http://dataservice.accuweather.com/locations/v1/cities/search"
                params = {"apikey": self.api_key, "q": location}
                async with session.get(location_url, params=params) as resp:
                    if resp.status != 200:
                        return {"error": f"Location API error {resp.status}", "location": location}
                    locations = await resp.json()
                    if not locations:
                        return {"error": "Location not found", "location": location}
                    location_key = locations[0]["Key"]
                current_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}"
                async with session.get(current_url, params={"apikey": self.api_key}) as resp:
                    if resp.status != 200:
                        return {"error": f"Weather API error {resp.status}", "location": location}
                    conditions = await resp.json()
                    if not conditions:
                        return {"error": "No weather data", "location": location}
                    current = conditions[0]
                    return {
                        "location": location,
                        "temperature": f"{current['Temperature']['Metric']['Value']}°C",
                        "conditions": current["WeatherText"]
                    }
        except Exception as e:
            return {"error": str(e), "location": location}

mcp = FastMCP("Weather Agent", port=8182, debug=True , log_level="DEBUG", host="0.0.0.0")

@mcp.tool()
async def get_weather(location: str = "Brussels"):
    return await WeatherMCPServer().get_weather(location)

@mcp.tool()
async def health_check():
    return {"status": "healthy", "api_key_configured": True}

if __name__ == "__main__":
    mcp.run(transport="sse")  # Use "sse" for web clients, "stdio" for CLI