import os
import logging
import sys
import json
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi_health import health
from prometheus_client import start_http_server, Counter, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from threading import Thread

# --- Logging ---
logger = logging.getLogger("weather-mcp")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
def log_structured(event, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))

# --- Metrics ---
REQUESTS = Counter('weather_mcp_requests_total', 'Total weather requests')
ERRORS = Counter('weather_mcp_errors_total', 'Total weather errors')
HEALTH = Gauge('weather_mcp_health', 'Weather MCP health status')

# --- Config ---
OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY", "7c1eca57030b77b41c31fe8c1fc210f6")
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
MCP_PROXY_URL = os.getenv("MCP_PROXY_URL", "http://mcp-proxy:8180")
SERVICE_PORT = int(os.getenv("WEATHER_MCP_PORT", "5000"))






# --- FastAPI setup ---
app = FastAPI()

def healthy():
    return True

app.add_api_route("/health", health([healthy]))

@app.get("/metrics")
def metrics():
    return JSONResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/weather")
def get_weather(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    current_weather: bool = Query(True, description="Get current weather"),
    hourly: str = Query("temperature_2m,precipitation", description="Comma-separated hourly params")
):
    """
    Proxy to Open-Meteo API for current and hourly weather.
    """
    REQUESTS.inc()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true" if current_weather else "false",
        "hourly": hourly,
    }
    

    
    try:
        # Log the actual URL being called for debugging
        log_structured("weather_api_call", url=OPEN_METEO_BASE, params=params)
        
        r = requests.get(OPEN_METEO_BASE, params=params, timeout=10)
        r.raise_for_status()
        log_structured("weather_query", latitude=latitude, longitude=longitude, status="success")
        return r.json()
    except requests.exceptions.HTTPError as e:
        ERRORS.inc()
        log_structured("weather_query_error", 
                      latitude=latitude, 
                      longitude=longitude, 
                      error=str(e),
                      status_code=e.response.status_code if e.response else None,
                      response_text=e.response.text if e.response else None)
        raise HTTPException(status_code=502, detail=f"Weather API error: {str(e)}")
    except Exception as e:
        ERRORS.inc()
        log_structured("weather_query_error", latitude=latitude, longitude=longitude, error=str(e))
        raise HTTPException(status_code=502, detail="Weather API unavailable")


# --- MCP Tool Registration ---
def register_with_mcp():
    """
    Register this weather tool with the MCP proxy.
    """
    tool_spec = {
        "name": "weather",
        "description": "Get current and forecast weather for a location.",
        "endpoint": f"http://weather-mcp:{SERVICE_PORT}/weather",
        "params": {
            "latitude": "float",
            "longitude": "float",
            "current_weather": "bool",
            "hourly": "str"
        }
    }
    # Add retry logic and better error handling
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First check if MCP proxy is reachable
            health_check = requests.get(f"{MCP_PROXY_URL}/health", timeout=5)
            log_structured("mcp_proxy_health", status_code=health_check.status_code)
            
            # Then try to register
            r = requests.post(f"{MCP_PROXY_URL}/register", json=tool_spec, timeout=5)
            r.raise_for_status()
            log_structured("mcp_register", status="success", tool="weather")
            return
        except Exception as e:
            log_structured("mcp_register_error", 
                          attempt=attempt + 1, 
                          max_retries=max_retries,
                          error=str(e))
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
    
    log_structured("mcp_register", status="failed_all_retries")

def start_metrics():
    start_http_server(9101)

def main():
    HEALTH.set(1)
    Thread(target=start_metrics, daemon=True).start()
    register_with_mcp()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)

if __name__ == "__main__":
    main()
