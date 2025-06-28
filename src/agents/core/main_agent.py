"""
Enhanced main agent system with proper AutoGen 2.x + LangGraph integration
"""
import asyncio
import time
import signal
import logging
import json
import uuid
from typing import Dict, Any, List, Optional
import typer
from fastapi import FastAPI, HTTPException
from prometheus_client import start_http_server, Counter
from autogen_core import MessageContext
from src.agents.core.orchestrator import orchestrator
from src.agents.specialized.weather_agent import ReliableWeatherAgent
from src.agents.specialized.routine_agent import ReliableRoutineAgent
from src.agents.core.database import db_manager
from config.settings import settings
from src.agents.core.a2a_protocol import a2a_registry
from src.agents.core.base_agent import agent_registry, AutoGenBaseAgent
from src.agents.core.logging_config import log_structured
from src.agents.core.agentic_workflow import HybridAgenticWorkflow
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger("jarvis")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Prometheus metrics
REQUESTS = Counter('jarvis_requests_total', 'Total requests')

class AutoGenLangGraphHybrid:
    """Hybrid system combining AutoGen agents with LangGraph workflows - PROPER SINGLETON"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.specialized_agents = {}
        self.workflow_system = None
        self._initialize_agents()
        self._initialize_workflow()
        
        AutoGenLangGraphHybrid._initialized = True
        logger.info("AutoGen-LangGraph Hybrid initialized (singleton)")

    def _initialize_agents(self):
        """Initialize specialized AutoGen agents - ONLY ONCE"""
        try:
            # Initialize core orchestrator
            self.orchestrator = orchestrator
            
            # Initialize specialized agents
            self.weather_agent = ReliableWeatherAgent()
            self.routine_agent = ReliableRoutineAgent()
            
            self.specialized_agents = {
                "weather": self.weather_agent,
                "routine": self.routine_agent,
                "orchestrator": self.orchestrator
            }
            
            logger.info(f"Initialized {len(self.specialized_agents)} specialized agents (singleton)")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise

    def _initialize_workflow(self):
        """Initialize the LangGraph workflow system - ONLY ONCE"""
        try:
            if not self.workflow_system:
                self.workflow_system = HybridAgenticWorkflow(
                    database=db_manager,
                    autogen_agents=self.specialized_agents
                )
                logger.info("LangGraph workflow system initialized (singleton)")
        except Exception as e:
            logger.error(f"Workflow initialization failed: {e}")
            raise

    async def process_message(self, message: str, thread_id: str = None) -> str:
        """Process message through the hybrid workflow - FIXED STATE CONVERSION"""
        try:
            thread_id = thread_id or str(uuid.uuid4())
            
            # CRITICAL FIX: Create dict input that matches HybridAgentState TypedDict
            workflow_input = {
                "session_id": thread_id,
                "agent_id": "hybrid_main",
                "user_input": message
            }
            
            # FIXED: Call the workflow with dict input (not ReasoningState)
            final_result = await self.workflow_system.run(workflow_input)
            
            logger.info(f"Workflow result type: {type(final_result)}")
            logger.info(f"Final result keys: {list(final_result.keys()) if isinstance(final_result, dict) else 'Not a dict'}")
            
            # IMPROVED: Extract response from dict result
            final_response = final_result.get("final_response", "") if isinstance(final_result, dict) else ""
            
            if final_response and final_response.strip():
                logger.info(f"Using final response: {final_response[:100]}")
                return final_response.strip()
            
            # Extract final response from dict result
            observations = final_result.get("observations", []) if isinstance(final_result, dict) else []
            successful_obs = [obs for obs in observations if obs.get("success", False)]
            
            if successful_obs:
                for obs in successful_obs:
                    result = obs.get("result", "")
                    if result and str(result).strip() and not str(result).startswith("Interaction stored"):
                        logger.info(f"Using observation result: {str(result)[:100]}")
                        return str(result).strip()
            
            # Fallback to observation results
            input_lower = message.lower()
            if any(word in input_lower for word in ["hello", "hi", "hey"]):
                return "Hello! I'm Jarvis, your AI assistant. How can I help you?"
            elif any(word in input_lower for word in ["weather", "temperature"]):
                return "I'd be happy to help with weather information. Could you specify a location?"
            else:
                return "I understand you're asking me something. How can I help you?"
                
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            return f"I encountered an error: {str(e)}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            db_health = db_manager.health_check()
            
            return {
                "status": "healthy",
                "system_type": "AutoGen-LangGraph Hybrid (Singleton)",
                "agents": {
                    "count": len(self.specialized_agents),
                    "types": list(self.specialized_agents.keys())
                },
                "workflow": {
                    "initialized": self.workflow_system is not None,
                    "type": "LangGraph"
                },
                "database": db_health.get("status", "unknown"),
                "a2a_agents": len(a2a_registry.agents),
                "timestamp": time.time(),
                "singleton_status": "initialized" if self._initialized else "not_initialized"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

class JarvisMainAgent(AutoGenBaseAgent):
    """Main agent that uses the hybrid system - SESSION AWARE"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AutoGenBaseAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_agent_initialized'):
            return
            
        super().__init__(
            name="jarvis_main",
            description="Main Jarvis agent with AutoGen-LangGraph hybrid processing",
            agent_type="hybrid_main"
        )
        
        self.hybrid_system = hybrid_system
        self._agent_initialized = True

    async def agentic_chat(self, message: str, thread_id: str = None) -> str:
        """Enhanced agentic chat using hybrid system"""
        return await self.hybrid_system.process_message(message, thread_id)

    async def handle_message(self, message: str, ctx: MessageContext) -> str:
        """AutoGen message handler"""
        return await self.agentic_chat(message)

# Initialize the hybrid system (singleton)
hybrid_system = AutoGenLangGraphHybrid()

# CLI Application
app = typer.Typer()

@app.command()
def agentic_chat():
    """Interactive agentic chat with AutoGen-LangGraph hybrid"""
    print("🤖 Jarvis Hybrid Chat (AutoGen + LangGraph)")
    print("Type 'quit' to exit")
    
    async def chat_loop():
        agent = JarvisMainAgent()
        thread_id = str(uuid.uuid4())
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                if user_input.lower() in ["quit", "exit"]:
                    break
                
                if user_input:
                    print("🤔 Processing with hybrid workflow...", end="", flush=True)
                    response = await agent.agentic_chat(user_input, thread_id)
                    print(f"\r🤖 Jarvis: {response}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    asyncio.run(chat_loop())

@app.command()
def chat(message: str):
    """Single chat message with hybrid system"""
    print(f"💬 You: {message}")
    
    async def single_chat():
        agent = JarvisMainAgent()
        response = await agent.agentic_chat(message)
        print(f"🤖 Jarvis: {response}")
    
    asyncio.run(single_chat())

@app.command()
def status():
    """Show comprehensive system status"""
    try:
        status_info = hybrid_system.get_system_status()
        print(f"🟢 System Status: {status_info['status']}")
        print(f"🏗️  System Type: {status_info['system_type']}")
        print(f"🤖 Agents: {status_info['agents']['count']} ({', '.join(status_info['agents']['types'])})")
        print(f"⚡ Workflow: {'Ready' if status_info['workflow']['initialized'] else 'Not Ready'}")
        print(f"💾 Database: {status_info['database']}")
        print(f"🔗 A2A Agents: {status_info['a2a_agents']}")
    except Exception as e:
        print(f"❌ Status check failed: {e}")

@app.command()
def serve():
    """Start the hybrid Jarvis service with full API"""
    # Start metrics server
    start_http_server(8001)
    print("📊 Metrics server: http://localhost:8001")
    
    # FastAPI application
    app_api = FastAPI(title="Jarvis Hybrid System", version="5.0.0")

    @app_api.post("/chat")
    async def chat_api(request: Dict[str, Any]):
        message = request.get("message", "")
        thread_id = request.get("thread_id")
        
        if not message:
            return {"error": "Message required"}
        
        agent = JarvisMainAgent()
        response = await agent.agentic_chat(message, thread_id)
        
        return {
            "response": response,
            "thread_id": thread_id or str(uuid.uuid4()),
            "system": "hybrid_singleton"
        }

    @app_api.get("/health")
    async def health():
        return hybrid_system.get_system_status()

    @app_api.get("/agents")
    async def list_agents():
        """List all available agents"""
        return {
            "autogen_agents": list(hybrid_system.specialized_agents.keys()),
            "a2a_agents": len(a2a_registry.agents),
            "total": len(hybrid_system.specialized_agents)
        }

    # Start FastAPI server
    import uvicorn
    import threading
    
    def run_api():
        uvicorn.run(app_api, host="0.0.0.0", port=8005, log_level="warning")
    
    threading.Thread(target=run_api, daemon=True).start()
    
    print("🚀 Jarvis Hybrid System started!")
    print("💬 Chat API: http://localhost:8005/chat")
    print("❤️  Health: http://localhost:8005/health")
    print("⚡ Workflow: AutoGen + LangGraph integration active")
    print("🔗 A2A Protocol: http://localhost:8005/.well-known/agent.json")
    
    # Service monitoring loop
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        print("\n🛑 Shutting down hybrid system...")
        running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    while running:
        try:
            # Update heartbeats for all agents
            for agent_id in hybrid_system.specialized_agents.keys():
                db_manager.update_agent_heartbeat(agent_id)
            
            # Log system status
            status = hybrid_system.get_system_status()
            if status["status"] == "healthy":
                print("✅ Jarvis Hybrid System operational")
            else:
                print(f"⚠️  System status: {status.get('error', 'unknown issue')}")
            
            time.sleep(30)
            
        except Exception as e:
            print(f"⚠️  Monitoring error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    app()
