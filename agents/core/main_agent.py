import asyncio
import time
import signal
import logging
import json
from typing import Dict, Any
import typer
from fastapi import FastAPI
from prometheus_client import start_http_server, Counter
from autogen_core import MessageContext
from agents.core.orchestrator import orchestrator
from agents.specialized.weather_agent import ReliableWeatherAgent
from agents.specialized.routine_agent import ReliableRoutineAgent
from agents.core.database import db_manager
from config.settings import settings

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("jarvis")

# Prometheus metrics
REQUESTS = Counter('jarvis_requests_total', 'Total requests')

class SimpleMessageContext:
    """Simple context for AutoGen messages"""
    def __init__(self, sender: str = "user"):
        self.sender = sender

class AutoGenJarvis:
    """Simple AutoGen-based Jarvis interface"""
    
    def __init__(self):
        self.orchestrator = orchestrator
        logger.info("✅ AutoGen Jarvis initialized")
    
    async def chat(self, message: str) -> str:
        """Simple chat interface using AutoGen"""
        REQUESTS.inc()
        
        try:
            # Use AutoGen's native message handling
            context = SimpleMessageContext("user")
            response = await self.orchestrator.handle_user_message(message, context)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Chat failed: {e}")
            return "I'm having trouble right now. Please try again."

# Initialize Jarvis
jarvis = AutoGenJarvis()

# CLI Commands
app = typer.Typer()

@app.command()
def chat(message: str):
    """Chat with AutoGen Jarvis"""
    print(f"🧠 You: {message}")
    
    # Run async chat
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(jarvis.chat(message))
    loop.close()
    
    print(f"🤖 Jarvis: {response}")

@app.command() 
def status():
    """Show system status"""
    try:
        health = db_manager.health_check()
        print(f"🏥 Database: {health.get('status', 'unknown')}")
        print(f"🤖 Orchestrator: Active")
        print(f"📊 Available agents: weather, routine")
    except Exception as e:
        print(f"❌ Status check failed: {e}")

@app.command()
def serve():
    """Start AutoGen Jarvis service"""
    # Start metrics
    start_http_server(8001)
    print("📊 Metrics server: http://localhost:8001")
    
    # FastAPI
    app_api = FastAPI(title="AutoGen Jarvis", version="4.0.0")
    
    @app_api.post("/chat")
    async def chat_api(request: Dict[str, Any]):
        message = request.get("message", "")
        if not message:
            return {"error": "Message required"}
        
        response = await jarvis.chat(message)
        return {"response": response}
    
    @app_api.get("/health")
    async def health():
        return {"status": "healthy", "system": "AutoGen Jarvis"}
    
    # Start FastAPI
    import uvicorn
    import threading
    
    def run_api():
        uvicorn.run(app_api, host="0.0.0.0", port=8005, log_level="warning")
    
    threading.Thread(target=run_api, daemon=True).start()
    
    print("🚀 AutoGen Jarvis started!")
    print("💬 Chat API: http://localhost:8005/chat")
    print("❤️  Health: http://localhost:8005/health")
    
    # Service loop
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("🛑 Shutting down...")
        running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    while running:
        try:
            db_manager.update_agent_heartbeat("orchestrator_agent")
            print("💚 AutoGen Jarvis operational")
            time.sleep(30)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    app()
