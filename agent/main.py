import os
import sys
import signal
import time
import json
import logging
from typing import Optional, Dict, Any
import typer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi_health import health as fastapi_health_check
from prometheus_client import start_http_server, Counter, Gauge
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from autogen import Agent, GroupChat, GroupChatManager  # Microsoft AutoGen 2.x
import langgraph
from apscheduler.schedulers.background import BackgroundScheduler
from sktime.forecasting.naive import NaiveForecaster
import wittgenstein as lw
import pandas as pd

# --- Logging Setup ---
logger = logging.getLogger("jarvis")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
def log_structured(event, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))

# --- Metrics ---
REQUESTS = Counter('agent_requests_total', 'Total requests to the agent')
ROUTINES_PROPOSED = Counter('agent_routines_proposed_total', 'Candidate routines proposed')
ROUTINES_APPROVED = Counter('agent_routines_approved_total', 'Candidate routines approved')
AGENT_HEALTH = Gauge('agent_health', 'Agent health status')

# --- Environment ---
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://jarvis:strongpassword@postgres:5432/jarvisdb")
CHROMA_URL = os.getenv("CHROMA_URL", "http://chroma:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MCP_PROXY_URL = os.getenv("MCP_PROXY_URL", "http://mcp-proxy:8180")
OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY", "")

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


# --- Chroma Semantic Memory ---
def add_semantic_memory(embedding, content, meta=None):
    # try:
    #     conn = get_pg_conn()
    #     with conn, conn.cursor() as cur:
    #         cur.execute(
    #             "INSERT INTO semantic_memory (embedding, content, meta) VALUES (%s, %s, %s)",
    #             (embedding, content, json.dumps(meta) if meta else None)
    #         )
    #     conn.close()
    #     return True
    # except Exception as e:
    #     log_event("ERROR", "Failed to add semantic memory", {"error": str(e)})
    #     return False
    try:
        r = requests.post(f"{CHROMA_URL}/api/v2/collections/default/documents", json={
            "embedding": embedding,
            "content": content,
            "meta": meta or {}
        })
        return r.ok
    except Exception as e:
        log_event("ERROR", "Failed to add semantic memory", {"error": str(e)})
        return False

def search_semantic_memory(query_embedding, limit=5):
    try:
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT content, meta, (embedding <=> %s) as distance 
                FROM semantic_memory 
                ORDER BY distance 
                LIMIT %s
            """, (query_embedding, limit))
            results = cur.fetchall()
        conn.close()
        return results
    except Exception as e:
        log_event("ERROR", "Semantic search failed", {"error": str(e)})
        return []


# --- Ollama Integration ---
def query_ollama(prompt: str, model: str = "llama3:latest"):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        r.raise_for_status()
        return r.json().get("response")
    except Exception as e:
        log_event("ERROR", "Ollama request failed", {"error": str(e)})
        return "LLM unavailable"

# --- MCP Proxy Integration ---
def call_mcp_tool(tool: str, params: Dict[str, Any]):
    try:
        r = requests.post(f"{MCP_PROXY_URL}/tools/{tool}", json=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log_event("ERROR", "MCP tool call failed", {"tool": tool, "error": str(e)})
        return {"error": "MCP unavailable"}

# --- Google A2A Integration (stub, expand as needed) ---
def register_a2a_tools():
    # Register agent's capabilities for agent-to-agent interop
    pass

# --- Behavior Learning Pipeline ---
def train_time_series_forecaster(y):
    forecaster = NaiveForecaster(strategy="mean")
    forecaster.fit(y)
    return forecaster

def learn_rules(X, y):
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
    ROUTINES_PROPOSED.inc()
    approved = user_approval_callback(candidate)
    if approved:
        ROUTINES_APPROVED.inc()
        log_event("INFO", "Routine approved", {"routine": candidate})
        return True
    else:
        log_event("INFO", "Routine rejected", {"routine": candidate})
        return False

# --- Typer CLI ---
app = typer.Typer()

@app.command()
def chat(message: str):
    """Chat with Jarvis AI."""
    REQUESTS.inc()
    log_event("INFO", "chat_invoked", {"message": message})
    response = query_ollama(message)
    print(f"🤖 Jarvis: {response}")

@app.command()
def routine(action: str = typer.Argument(..., help="train|propose|list|approve")):
    """Manage routines: train, propose, list, approve."""
    if action == "train":
        # Example: Train on synthetic time series
        from sktime.datasets import load_airline
        y = load_airline()
        model = train_time_series_forecaster(y)
        print("Routine trained (time series forecast):", model.predict([len(y), len(y)+1, len(y)+2]))
    elif action == "propose":
        candidate = {
            "name": "Morning Workout",
            "time": "07:00",
            "duration": "30 minutes",
            "activities": ["stretching", "cardio"]
        }
        def user_approval(candidate):
            print(f"Approve this routine? {candidate}")
            return True
        propose_routine(candidate, user_approval)
    elif action == "list":
        try:
            conn = get_pg_conn()
            with conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_logs WHERE message LIKE '%Routine approved%'")
                routines = cur.fetchall()
                print("Approved routines:", routines)
            conn.close()
        except Exception as e:
            print("Error listing routines:", str(e))
    elif action == "approve":
        print("Routine approval flow not implemented in CLI (use API/UI).")
    else:
        print("Unknown action.")

@app.command()
def logs(level: Optional[str] = None):
    """Show agent logs."""
    try:
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            if level:
                cur.execute("SELECT * FROM agent_logs WHERE level=%s ORDER BY timestamp DESC LIMIT 20", (level,))
            else:
                cur.execute("SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT 20")
            for row in cur.fetchall():
                print(row)
        conn.close()
    except Exception as e:
        print("Error fetching logs:", str(e))

@app.command()
def serve():
    """Run as a service (CLI and API)."""
    # Start Prometheus metrics server
    start_http_server(8001)
    print("📊 Prometheus metrics server started on port 8001")
    
    # Start FastAPI for health and metrics
    fastapi_app = FastAPI(title="Jarvis AI Agent", version="1.0.0")
    AGENT_HEALTH.set(1)
    
    # Define health check conditions that return dictionaries
    def check_database():
        try:
            conn = get_pg_conn()
            with conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
            conn.close()
            return {"database": "online", "status": "healthy"}
        except Exception as e:
            return False  # This will trigger failure status
    
    def check_ollama():
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                return {"ollama": "online", "status": "healthy"}
            return False
        except Exception:
            return False
    
    def check_chroma():
        try:
            r = requests.get(f"{CHROMA_URL}/api/v1/heartbeat", timeout=5)
            if r.status_code == 200:
                return {"chroma": "online", "status": "healthy"}
            return False
        except Exception:
            return False
    
    # Add health endpoint with multiple conditions
    fastapi_app.add_api_route(
        "/health", 
        fastapi_health_check([check_database, check_ollama, check_chroma])
    )
    
    # Add a simple status endpoint for debugging
    @fastapi_app.get("/status")
    async def get_status():
        return {
            "service": "Jarvis AI Agent",
            "status": "running",
            "version": "1.0.0",
            "metrics_port": 8001,
            "health_port": 8005,
            "timestamp": time.time()
        }
    
    @fastapi_app.get("/metrics")
    def metrics():
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    # Start FastAPI in background thread
    import uvicorn
    import threading
    
    def run_api():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8005, log_level="info")
    
    threading.Thread(target=run_api, daemon=True).start()
    
    print("🔗 Health endpoint: http://0.0.0.0:8005/health")
    print("📊 Metrics endpoint: http://0.0.0.0:8001/metrics")
    print("📊 Status endpoint: http://0.0.0.0:8005/status")
    
    # Rest of your serve function...
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("🛑 Jarvis AI Agent shutting down gracefully...")
        AGENT_HEALTH.set(0)
        running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Jarvis AI Agent service started.")
    while running:
        print("💚 Agent heartbeat - all systems operational")
        time.sleep(10)
    print("✅ Jarvis AI Agent service stopped.")


if __name__ == "__main__":
    app()
