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
from agents.core.a2a_protocol import a2a_registry
from fastapi import Request
import uuid


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
        # Initialize specialized agents (this will trigger A2A registration)
        self._initialize_agents()
        logger.info("✅ AutoGen Jarvis initialized")
        logger.info(f"🤖 A2A agents registered: {len(a2a_registry.agents)}")

    def _initialize_agents(self):
        """Initialize specialized agents with A2A registration"""
        try:
            # Import and initialize agents (this triggers A2A registration)
            from agents.specialized.weather_agent import ReliableWeatherAgent
            from agents.specialized.routine_agent import ReliableRoutineAgent
            
            # Create instances (A2A registration happens in __init__)
            weather_agent = ReliableWeatherAgent()
            routine_agent = ReliableRoutineAgent()
            
            logger.info("✅ Specialized agents initialized with A2A support")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
    
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
    """Start AutoGen Jarvis service with A2A protocol support"""
    # Start metrics
    start_http_server(8001)
    print("📊 Metrics server: http://localhost:8001")
    
    # FastAPI
    app_api = FastAPI(title="AutoGen Jarvis with A2A", version="4.0.0")
    
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
    
    # A2A Protocol Endpoints
    @app_api.get("/.well-known/agent.json")
    async def agent_discovery():
        """A2A agent discovery endpoint"""
        agents = a2a_registry.discover_agents()
        return {
            "agents": [agent.to_dict() for agent in agents],
            "registry_info": {
                "total_agents": len(agents),
                "protocol_version": "1.0",
                "supported_methods": ["request_response", "sse", "push_notification"]
            }
        }
    
    @app_api.get("/a2a/agents")
    async def list_a2a_agents():
        """List all A2A agents"""
        agents = a2a_registry.discover_agents()
        return {"agents": [agent.to_dict() for agent in agents]}
    
    @app_api.get("/a2a/agents/{agent_id}")
    async def get_a2a_agent(agent_id: str):
        """Get specific A2A agent card"""
        agent = a2a_registry.get_agent_card(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent.to_dict()
    
    @app_api.post("/a2a/agents/{agent_id}/request")
    async def a2a_agent_request(agent_id: str, request: Dict[str, Any]):
        """Send A2A request to specific agent"""
        # Get the target agent
        from agents.core.base_agent import agent_registry
        
        target_agent = agent_registry.get_agent(agent_id)
        if not target_agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if agent supports A2A
        if not hasattr(target_agent, 'handle_a2a_request'):
            raise HTTPException(status_code=400, detail="Agent does not support A2A protocol")
        
        # Process A2A request
        from_agent = request.get("from_agent", "external_client")
        task = request.get("task", {})
        
        result = await target_agent.handle_a2a_request(from_agent, task)
        return result
    
    @app_api.get("/a2a/agents/{agent_id}/health")
    async def a2a_agent_health(agent_id: str):
        """A2A agent health check"""
        agent = a2a_registry.get_agent_card(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent_id,
            "status": "healthy",
            "last_updated": agent.to_dict().get("last_updated"),
            "skills_count": len(agent.skills)
        }
    
    @app_api.get("/a2a/discover")
    async def a2a_discover(skill: str = None):
        """Discover A2A agents by skill"""
        agents = a2a_registry.discover_agents(skill)
        return {
            "query": {"skill_filter": skill},
            "agents": [agent.to_dict() for agent in agents],
            "count": len(agents)
        }
    
    @app_api.get("/a2a/communications")
    async def a2a_communications():
        """Get A2A communication logs"""
        return {
            "communications": a2a_registry.communication_log[-10:],  # Last 10
            "total_communications": len(a2a_registry.communication_log)
        }
    
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
