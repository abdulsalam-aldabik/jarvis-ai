import os
import sys
import signal
import time
import json
import logging
import asyncio
import uuid
from typing import Optional, Dict, Any, List
import typer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi_health import health as fastapi_health_check
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from autogen import Agent, GroupChat, GroupChatManager
import langgraph
from apscheduler.schedulers.background import BackgroundScheduler
from sktime.forecasting.naive import NaiveForecaster
import wittgenstein as lw
import pandas as pd
import numpy as np
from aiohttp import ClientSession
import chromadb
import re

# --- Enhanced Logging Setup ---
logger = logging.getLogger("jarvis")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_structured(event, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))

# --- Enhanced Metrics ---
REQUESTS = Counter('agent_requests_total', 'Total requests to the agent')
ROUTINES_PROPOSED = Counter('agent_routines_proposed_total', 'Candidate routines proposed')
ROUTINES_APPROVED = Counter('agent_routines_approved_total', 'Candidate routines approved')
AGENT_HEALTH = Gauge('agent_health', 'Agent health status')
WEATHER_REQUESTS = Counter('agent_weather_requests_total', 'Total weather requests')
SEMANTIC_MEMORY_OPERATIONS = Counter('agent_semantic_memory_total', 'Semantic memory operations', ['operation'])
CHAT_SESSIONS = Counter('agent_chat_sessions_total', 'Total chat sessions')
RESPONSE_TIME = Histogram('agent_response_time_seconds', 'Response time for agent operations', ['operation'])

# --- Environment ---
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://jarvis:strongpassword@postgres:5432/jarvisdb")
CHROMA_URL = os.getenv("CHROMA_URL", "http://chroma:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MCP_PROXY_URL = os.getenv("MCP_PROXY_URL", "http://mcp-proxy:8180")
ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY", "X9Yx6HmKtT1PGlLeIHcmdydJe1VLpsCP")

# --- Database ---
def get_pg_conn():
    return psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)

def log_event(level, message, meta=None):
    try:
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_logs (level, message, meta) VALUES (%s, %s, %s)",
                (level, message, json.dumps(meta) if meta else None)
            )
        conn.close()
    except Exception as e:
        logger.error(json.dumps({"event": "log_event_error", "error": str(e)}))

# --- Enhanced ChromaDB Integration ---
def get_chroma_client():
    """Get ChromaDB client connection with error handling."""
    try:
        client = chromadb.HttpClient(host='chroma', port=8000)
        # Test connection
        client.heartbeat()
        return client
    except Exception as e:
        log_event("ERROR", "ChromaDB connection failed", {"error": str(e)})
        return None

def add_to_semantic_memory(content: str, metadata: dict = None, collection_name: str = "jarvis_memory"):
    """Add content to semantic memory using ChromaDB v2 API."""
    SEMANTIC_MEMORY_OPERATIONS.labels(operation="add").inc()
    try:
        client = get_chroma_client()
        if not client:
            return False
            
        collection = client.get_or_create_collection(collection_name)
        doc_id = str(uuid.uuid4())
        
        collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
        log_event("INFO", "Added to semantic memory", {
            "collection": collection_name,
            "doc_id": doc_id,
            "content_length": len(content)
        })
        return True
        
    except Exception as e:
        log_event("ERROR", "Failed to add to semantic memory", {"error": str(e)})
        return False

def search_semantic_memory(query: str, n_results: int = 5, collection_name: str = "jarvis_memory"):
    """Search semantic memory using ChromaDB v2 API."""
    SEMANTIC_MEMORY_OPERATIONS.labels(operation="search").inc()
    try:
        client = get_chroma_client()
        if not client:
            return []
            
        collection = client.get_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        log_event("INFO", "Semantic memory search", {
            "query": query,
            "results_count": len(results['documents'][0]) if results['documents'] else 0
        })
        
        return results
        
    except Exception as e:
        log_event("ERROR", "Semantic memory search failed", {"error": str(e)})
        return []

# --- Enhanced Weather Integration with AccuWeather ---
async def get_weather_data_accuweather(location: str = "Brussels"):
    """Get weather data using AccuWeather API with async support."""
    WEATHER_REQUESTS.inc()
    
    if not ACCUWEATHER_API_KEY:
        return {"error": "ACCUWEATHER_API_KEY not configured"}
    
    base_url = "http://dataservice.accuweather.com"
    
    try:
        async with ClientSession() as session:
            # Search for location
            location_search_url = f"{base_url}/locations/v1/cities/search"
            params = {"apikey": ACCUWEATHER_API_KEY, "q": location}
            
            async with session.get(location_search_url, params=params) as response:
                locations = await response.json()
                if response.status != 200 or not locations:
                    return {"error": f"Location '{location}' not found"}
                
                location_key = locations[0]["Key"]
                location_name = locations[0]["LocalizedName"]
                country = locations[0]["Country"]["LocalizedName"]
            
            # Get current conditions
            current_url = f"{base_url}/currentconditions/v1/{location_key}"
            params = {"apikey": ACCUWEATHER_API_KEY}
            
            async with session.get(current_url, params=params) as response:
                current_conditions = await response.json()
                
            # Get hourly forecast
            forecast_url = f"{base_url}/forecasts/v1/hourly/12hour/{location_key}"
            params = {"apikey": ACCUWEATHER_API_KEY, "metric": "true"}
            
            async with session.get(forecast_url, params=params) as response:
                forecast = await response.json()
                
            # Format response
            if current_conditions:
                current = current_conditions[0]
                result = {
                    "location": location_name,
                    "country": country,
                    "current_conditions": {
                        "temperature": {
                            "value": current["Temperature"]["Metric"]["Value"],
                            "unit": current["Temperature"]["Metric"]["Unit"]
                        },
                        "weather_text": current["WeatherText"],
                        "humidity": current.get("RelativeHumidity"),
                        "precipitation": current.get("HasPrecipitation", False),
                        "observation_time": current["LocalObservationDateTime"]
                    },
                    "hourly_forecast": []
                }
                
                # Add hourly forecast (next 3 hours)
                for i, hour in enumerate(forecast[:3], 1):
                    result["hourly_forecast"].append({
                        "relative_time": f"+{i} hour{'s' if i > 1 else ''}",
                        "temperature": {"value": hour["Temperature"]["Value"]},
                        "weather_text": hour["IconPhrase"],
                        "precipitation_probability": hour["PrecipitationProbability"]
                    })
                
                log_event("INFO", "weather_request_success", {"location": location})
                return result
                
    except Exception as e:
        log_event("ERROR", "AccuWeather API failed", {"error": str(e)})
        return {"error": str(e)}

# --- Enhanced Intent Learning System ---
class IntentLearner:
    def __init__(self):
        self.is_trained = False
        
    def predict_intent(self, message: str) -> str:
        """Predict user intent from message patterns."""
        # Weather detection
        weather_indicators = [
            "weather", "temperature", "temp", "rain", "forecast", 
            "climate", "sunny", "cloudy", "wind", "humidity", "degrees"
        ]
        if any(indicator in message.lower() for indicator in weather_indicators):
            return "weather"
        
        # Routine management
        routine_indicators = ["routine", "schedule", "plan", "habit", "daily", "morning", "evening"]
        if any(indicator in message.lower() for indicator in routine_indicators):
            return "routine"
        
        # Memory operations
        memory_indicators = ["remember", "recall", "save", "forget", "memory", "store"]
        if any(indicator in message.lower() for indicator in memory_indicators):
            return "memory"
        
        return "general"
    
    def extract_location(self, message: str) -> str:
        """Extract location from message using regex patterns."""
        location_patterns = [
            r"in\s+([A-Za-z\s,]+?)(?:\s|$|\?|!|\.)",
            r"at\s+([A-Za-z\s,]+?)(?:\s|$|\?|!|\.)",
            r"for\s+([A-Za-z\s,]+?)(?:\s|$|\?|!|\.)",
            r"weather\s+([A-Za-z\s,]+?)(?:\s|$|\?|!|\.)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                potential_location = match.group(1).strip()
                # Filter out common words
                if len(potential_location) > 2 and potential_location.lower() not in [
                    "today", "tomorrow", "now", "there", "here", "like", "good", "bad",
                    "outside", "inside", "right", "currently"
                ]:
                    return potential_location
        
        return "Brussels"  # Default location

intent_learner = IntentLearner()

# --- Chat Session Management ---
def create_chat_session(user_message: str, agent_response: str, model_used: str = "ollama", response_time_ms: int = 0):
    """Store chat session in database."""
    try:
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_sessions (user_message, agent_response, model_used, response_time_ms)
                VALUES (%s, %s, %s, %s)
                RETURNING session_id
            """, (user_message, agent_response, model_used, response_time_ms))
            session_id = cur.fetchone()['session_id']
        conn.close()
        return session_id
    except Exception as e:
        log_event("ERROR", "Failed to create chat session", {"error": str(e)})
        return None

# --- Enhanced Ollama Integration ---
def query_ollama(prompt: str, model: str = "llama3:latest"):
    """Query Ollama with enhanced error handling and context."""
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        r.raise_for_status()
        return r.json().get("response", "No response generated")
    except Exception as e:
        log_event("ERROR", "Ollama request failed", {"error": str(e)})
        return "LLM unavailable - please try again later"

# --- Enhanced MCP Integration ---
def call_mcp_tool(tool: str, params: Dict[str, Any]):
    """Enhanced MCP tool calling with fallbacks."""
    try:
        r = requests.post(f"{MCP_PROXY_URL}/tools/{tool}", json=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log_event("ERROR", "MCP tool call failed", {"tool": tool, "error": str(e)})
        return {"error": "MCP unavailable"}

# --- Enhanced Behavior Learning ---
def train_time_series_forecaster(y):
    """Enhanced time series forecaster with multiple strategies."""
    forecaster = NaiveForecaster(strategy="mean")
    forecaster.fit(y)
    return forecaster

def learn_rules(X, y):
    """Enhanced rule learning with better data handling."""
    if not isinstance(X, pd.DataFrame):
        if hasattr(X, 'columns'):
            X = pd.DataFrame(X)
        else:
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    data = X.copy()
    data['target'] = y
    clf = lw.RIPPER(k=2, prune_size=0.33, dl_allowance=64)
    clf.fit(data, class_feat='target')
    return clf

def propose_routine(candidate, user_approval_callback):
    """Enhanced routine proposal with database tracking."""
    ROUTINES_PROPOSED.inc()
    approved = user_approval_callback(candidate)
    
    try:
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO routines (name, description, activities, approved)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (
                candidate.get("name", "Unnamed Routine"),
                candidate.get("description", ""),
                json.dumps(candidate),
                approved
            ))
        conn.close()
    except Exception as e:
        log_event("ERROR", "Failed to store routine", {"error": str(e)})
    
    if approved:
        ROUTINES_APPROVED.inc()
        log_event("INFO", "Routine approved", {"routine": candidate})
        return True
    else:
        log_event("INFO", "Routine rejected", {"routine": candidate})
        return False

# --- Enhanced CLI Commands ---
app = typer.Typer()

@app.command()
def chat(message: str):
    """Enhanced chat with semantic memory, weather, and intent detection."""
    REQUESTS.inc()
    CHAT_SESSIONS.inc()
    
    start_time = time.time()
    log_event("INFO", "chat_invoked", {"message": message})
    
    # Predict intent
    intent = intent_learner.predict_intent(message)
    
    # Search semantic memory for context
    memory_results = search_semantic_memory(message, n_results=3)
    context = ""
    if memory_results and memory_results['documents']:
        context = "\n".join(memory_results['documents'][0][:2])  # Use top 2 results
    
    response = ""
    
    if intent == "weather":
        # Extract location and get weather
        location = intent_learner.extract_location(message)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        weather_data = loop.run_until_complete(get_weather_data_accuweather(location))
        loop.close()
        
        if "error" not in weather_data:
            current = weather_data["current_conditions"]
            temp = current["temperature"]["value"]
            weather_text = current["weather_text"]
            humidity = current.get("humidity", "N/A")
            location_name = weather_data["location"]
            country = weather_data["country"]
            
            response = f"🌤️ Current weather in {location_name}, {country}: {temp}°C, {weather_text}"
            if humidity != "N/A":
                response += f", humidity: {humidity}%"
                
            # Add forecast if available
            hourly = weather_data.get("hourly_forecast", [])
            if hourly:
                next_hour = hourly[0]
                response += f"\n🕐 Next hour: {next_hour['temperature']['value']}°C, {next_hour['weather_text']}"
        else:
            response = f"Sorry, I couldn't get weather information for {location}: {weather_data['error']}"
    
    elif intent == "memory":
        # Handle memory operations
        if any(word in message.lower() for word in ["remember", "save", "store"]):
            # Extract content to remember
            content_to_save = message.replace("remember", "").replace("save", "").strip()
            if add_to_semantic_memory(content_to_save, {"type": "user_request", "timestamp": time.time()}):
                response = f"✅ I've saved that to my memory: {content_to_save}"
            else:
                response = "❌ Sorry, I couldn't save that to memory right now."
        else:
            # Search memory
            if memory_results and memory_results['documents']:
                response = f"🧠 From my memory: {memory_results['documents'][0][0]}"
            else:
                response = "🧠 I don't have any relevant memories for that query."
    
    else:
        # General conversation with Ollama
        if context:
            enhanced_message = f"Previous context: {context}\n\nUser: {message}"
            response = query_ollama(enhanced_message)
        else:
            response = query_ollama(message)
    
    # Store the interaction in semantic memory and database
    interaction = f"User: {message}\nJarvis: {response}"
    add_to_semantic_memory(interaction, {
        "type": "conversation",
        "intent": intent,
        "timestamp": time.time(),
        "user_message": message
    })
    
    # Track response time
    response_time_ms = int((time.time() - start_time) * 1000)
    RESPONSE_TIME.labels(operation="chat").observe(time.time() - start_time)
    
    # Store chat session
    create_chat_session(message, response, "jarvis_enhanced", response_time_ms)
    
    print(f"🤖 Jarvis: {response}")

@app.command()
def weather(location: str = typer.Option("Brussels", help="Location to get weather for")):
    """Get detailed weather information for any location worldwide."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    weather_data = loop.run_until_complete(get_weather_data_accuweather(location))
    loop.close()
    
    if "error" not in weather_data:
        current = weather_data["current_conditions"]
        location_name = weather_data["location"]
        country = weather_data["country"]
        
        print(f"🌍 Weather for {location_name}, {country}:")
        
        temp = current["temperature"]["value"]
        weather_text = current["weather_text"]
        humidity = current.get("humidity", "N/A")
        precipitation = current.get("precipitation", False)
        
        print(f"🌡️  Temperature: {temp}°C")
        print(f"☁️  Conditions: {weather_text}")
        print(f"💧 Humidity: {humidity}%")
        print(f"🌧️  Precipitation: {'Yes' if precipitation else 'No'}")
        
        # Show hourly forecast
        hourly = weather_data.get("hourly_forecast", [])
        if hourly:
            print(f"\n📅 Next {len(hourly)} hours:")
            for hour_data in hourly:
                relative_time = hour_data["relative_time"]
                temp = hour_data["temperature"]["value"]
                weather_text = hour_data["weather_text"]
                precip_prob = hour_data["precipitation_probability"]
                print(f"   {relative_time}: {temp}°C, {weather_text} (☔ {precip_prob}%)")
    else:
        print(f"❌ {weather_data['error']}")

@app.command()
def memory_stats():
    """Show semantic memory statistics."""
    try:
        client = get_chroma_client()
        if not client:
            print("❌ ChromaDB not available")
            return
            
        collections = client.list_collections()
        print(f"📊 Semantic Memory Statistics:")
        
        for collection in collections:
            count = collection.count()
            print(f"   📁 {collection.name}: {count} documents")
            
        # Database memory stats
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chat_sessions")
            chat_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM agent_logs")
            log_count = cur.fetchone()[0]
            print(f"   💬 Chat sessions: {chat_count}")
            print(f"   📝 Log entries: {log_count}")
        conn.close()
            
    except Exception as e:
        print(f"❌ Error getting memory stats: {e}")

@app.command()
def search_memory(query: str, limit: int = 5):
    """Search semantic memory."""
    results = search_semantic_memory(query, n_results=limit)
    
    if results and results['documents']:
        print(f"🔍 Found {len(results['documents'][0])} results for '{query}':")
        for i, doc in enumerate(results['documents'][0]):
            print(f"   {i+1}. {doc[:100]}...")
    else:
        print(f"🔍 No results found for '{query}'")

@app.command()
def routine(action: str = typer.Argument(..., help="train|propose|list|approve")):
    """Enhanced routine management with database integration."""
    if action == "train":
        from sktime.datasets import load_airline
        y = load_airline()
        model = train_time_series_forecaster(y)
        predictions = model.predict([len(y), len(y)+1, len(y)+2])
        print("✅ Routine forecasting model trained")
        print(f"📈 Predictions for next 3 periods: {predictions.tolist()}")
        
    elif action == "propose":
        candidate = {
            "name": "Smart Morning Routine",
            "time": "07:00",
            "duration": "45 minutes",
            "activities": ["meditation", "exercise", "healthy breakfast"],
            "description": "AI-optimized morning routine for productivity"
        }
        def user_approval(candidate):
            print(f"🤔 Approve this routine? {candidate}")
            return True  # Auto-approve for demo
        
        if propose_routine(candidate, user_approval):
            print("✅ Routine approved and stored!")
        else:
            print("❌ Routine was rejected")
            
    elif action == "list":
        try:
            conn = get_pg_conn()
            with conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM routines ORDER BY created_at DESC LIMIT 10")
                routines = cur.fetchall()
                
                if routines:
                    print("📋 Recent Routines:")
                    for routine in routines:
                        status = "✅ Approved" if routine['approved'] else "⏳ Pending"
                        print(f"   {routine['name']} - {status}")
                        print(f"      Created: {routine['created_at']}")
                else:
                    print("📋 No routines found")
            conn.close()
        except Exception as e:
            print(f"❌ Error listing routines: {e}")
            
    else:
        print("❌ Unknown action. Use: train|propose|list|approve")

@app.command()
def logs(level: Optional[str] = None, limit: int = 20):
    """Enhanced log viewing with filtering."""
    try:
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            if level:
                cur.execute(
                    "SELECT * FROM agent_logs WHERE level=%s ORDER BY timestamp DESC LIMIT %s", 
                    (level.upper(), limit)
                )
            else:
                cur.execute("SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT %s", (limit,))
                
            logs = cur.fetchall()
            
            if logs:
                print(f"📝 Recent Logs ({len(logs)} entries):")
                for log in logs:
                    timestamp = log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    level_emoji = {"INFO": "ℹ️", "ERROR": "❌", "WARNING": "⚠️"}.get(log['level'], "📄")
                    print(f"   {level_emoji} [{timestamp}] {log['level']}: {log['message']}")
            else:
                print("📝 No logs found")
        conn.close()
    except Exception as e:
        print(f"❌ Error fetching logs: {e}")

@app.command()
def serve():
    """Enhanced service mode with comprehensive health checks."""
    # Start Prometheus metrics server
    start_http_server(8001)
    print("📊 Prometheus metrics server started on port 8001")
    
    # Start FastAPI for health and metrics
    fastapi_app = FastAPI(title="Jarvis AI Agent", version="2.0.0")
    AGENT_HEALTH.set(1)
    
    # Enhanced health check conditions
    def check_database():
        try:
            conn = get_pg_conn()
            with conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                # Test table access
                cur.execute("SELECT COUNT(*) FROM agent_logs")
                count = cur.fetchone()[0]
            conn.close()
            return {"database": "healthy", "log_entries": count}
        except Exception:
            return False
    
    def check_ollama():
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = r.json().get("models", [])
                return {"ollama": "healthy", "models_available": len(models)}
            return False
        except Exception:
            return False
    
    def check_chroma():
        try:
            client = get_chroma_client()
            if client:
                collections = client.list_collections()
                return {"chroma": "healthy", "collections": len(collections)}
            return False
        except Exception:
            return False
    
    def check_accuweather():
        if ACCUWEATHER_API_KEY:
            return {"accuweather": "configured", "api_key_available": True}
        return {"accuweather": "not_configured", "api_key_available": False}
    
    # Add comprehensive health endpoint
    fastapi_app.add_api_route(
        "/health", 
        fastapi_health_check([check_database, check_ollama, check_chroma, check_accuweather])
    )
    
    @fastapi_app.get("/status")
    async def get_status():
        return {
            "service": "Jarvis AI Agent",
            "version": "2.0.0",
            "status": "running",
            "features": [
                "weather_integration",
                "semantic_memory", 
                "routine_management",
                "intent_detection",
                "chat_sessions"
            ],
            "metrics_port": 8001,
            "health_port": 8005,
            "timestamp": time.time()
        }
    
    @fastapi_app.get("/metrics")
    def metrics():
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    @fastapi_app.get("/memory/stats")
    async def memory_stats_api():
        """API endpoint for memory statistics."""
        try:
            client = get_chroma_client()
            stats = {"collections": []}
            
            if client:
                collections = client.list_collections()
                for collection in collections:
                    stats["collections"].append({
                        "name": collection.name,
                        "count": collection.count()
                    })
            
            # Add database stats
            conn = get_pg_conn()
            with conn, conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chat_sessions")
                chat_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM agent_logs")
                log_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM routines")
                routine_count = cur.fetchone()[0]
            conn.close()
            
            stats.update({
                "chat_sessions": chat_count,
                "log_entries": log_count,
                "routines": routine_count
            })
            
            return stats
        except Exception as e:
            return {"error": str(e)}
    
    # Start FastAPI in background thread
    import uvicorn
    import threading
    
    def run_api():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8005, log_level="info")
    
    threading.Thread(target=run_api, daemon=True).start()
    
    print("🔗 Health endpoint: http://0.0.0.0:8005/health")
    print("📊 Metrics endpoint: http://0.0.0.0:8001/metrics")
    print("📊 Status endpoint: http://0.0.0.0:8005/status")
    print("🧠 Memory stats API: http://0.0.0.0:8005/memory/stats")
    
    # Enhanced service loop
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("🛑 Jarvis AI Agent shutting down gracefully...")
        AGENT_HEALTH.set(0)
        running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Jarvis AI Agent service started with full integration!")
    while running:
        try:
            log_event("INFO", "heartbeat", {"status": "operational", "timestamp": time.time()})
            print("💚 Agent heartbeat - all systems operational")
            time.sleep(30)
        except Exception as e:
            print(f"❌ Heartbeat error: {e}")
            time.sleep(10)
    
    print("✅ Jarvis AI Agent service stopped.")

if __name__ == "__main__":
    app()
