#!/usr/bin/env python3
"""
SIMPLE: Direct HTTP weather server - no proxy needed
"""
import json
import os
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Weather Server", version="1.0.0")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEMO_MODE = not OPENWEATHER_API_KEY
AGENT_ID = "simple_weather_server"

# Request models
class WeatherRequest(BaseModel):
    location: str = "Brussels"

class ForecastRequest(BaseModel):
    location: str = "Brussels"
    days: int = 3

def get_demo_weather(location: str):
    """Demo weather data"""
    demo_data = {
        "brussels": {"temp": 18, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp": 15, "condition": "Rainy", "humidity": 80},
        "paris": {"temp": 20, "condition": "Sunny", "humidity": 55},
    }
    
    weather = demo_data.get(location.lower(), {"temp": 20, "condition": "Mild", "humidity": 60})
    
    return {
        "location": location.title(),
        "temperature": f"{weather['temp']}°C",
        "conditions": weather["condition"],
        "humidity": f"{weather['humidity']}%",
        "timestamp": datetime.now().isoformat(),
        "demo": True,
        "source": "demo_weather",
        "agent_id": AGENT_ID
    }

async def get_real_weather(location: str):
    """Get real weather from OpenWeatherMap"""
    try:
        params = {
            "q": location,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(
            "http://api.openweathermap.org/data/2.5/weather", 
            params=params, 
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        return {
            "location": data["name"],
            "country": data["sys"]["country"],
            "temperature": f"{data['main']['temp']}°C",
            "feels_like": f"{data['main']['feels_like']}°C",
            "conditions": data["weather"][0]["description"].title(),
            "humidity": f"{data['main']['humidity']}%",
            "pressure": f"{data['main']['pressure']} hPa",
            "wind_speed": f"{data.get('wind', {}).get('speed', 0)} m/s",
            "timestamp": datetime.now().isoformat(),
            "demo": False,
            "source": "openweathermap_api",
            "agent_id": AGENT_ID
        }
        
    except Exception as e:
        logger.error(f"OpenWeatherMap API failed: {e}")
        return None

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "agent_id": AGENT_ID,
        "demo_mode": DEMO_MODE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/weather")
async def get_weather(request: WeatherRequest):
    """Get current weather - simple endpoint"""
    try:
        location = request.location.strip() or "Brussels"
        
        logger.info(f"Weather request for: {location} (Demo: {DEMO_MODE})")
        
        # Try real API first
        if not DEMO_MODE:
            weather_data = await get_real_weather(location)
            if weather_data:
                logger.info(f"Real weather data retrieved for {location}")
                return weather_data
        
        # Fallback to demo
        logger.info(f"Using demo weather data for {location}")
        return get_demo_weather(location)
        
    except Exception as e:
        logger.error(f"Weather request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Weather request failed: {str(e)}")

@app.post("/forecast")
async def get_forecast(request: ForecastRequest):
    """Get weather forecast"""
    try:
        location = request.location.strip() or "Brussels"
        days = max(1, min(request.days, 7))
        
        # Generate demo forecast
        conditions = ["Sunny", "Partly cloudy", "Rainy", "Overcast", "Clear"]
        temps = [18, 21, 16, 19, 22]
        
        forecast_days = []
        for i in range(days):
            day_data = {
                "day": i + 1,
                "date": (datetime.now() + timedelta(days=i)).date().isoformat(),
                "temperature_high": f"{temps[i % len(temps)] + 3}°C",
                "temperature_low": f"{temps[i % len(temps)] - 2}°C",
                "conditions": conditions[i % len(conditions)],
                "humidity": f"{60 + (i * 5) % 30}%"
            }
            forecast_days.append(day_data)
        
        return {
            "location": location.title(),
            "forecast_days": days,
            "forecast": forecast_days,
            "timestamp": datetime.now().isoformat(),
            "demo": True,
            "agent_id": AGENT_ID
        }
        
    except Exception as e:
        logger.error(f"Forecast request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast request failed: {str(e)}")

if __name__ == "__main__":
    logger.info(f"🌤️ Starting Simple Weather Server (API Mode: {'Real' if not DEMO_MODE else 'Demo'})")
    uvicorn.run(app, host="0.0.0.0", port=8184)
