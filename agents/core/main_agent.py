import os
import sys
import signal
import time
import json
import logging
import asyncio
import uuid
from typing import Optional, Dict, Any
import typer
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi_health import health as fastapi_health_check
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# FIX: Import AgentTask properly
from agents.core.base_agent import AgentTask, IntelligentLLMInterface
from agents.core.orchestrator import orchestrator
from agents.specialized.weather_agent import WeatherAgent
from agents.specialized.routine_agent import RoutineAgent
from agents.core.database import db_manager
from config.settings import settings
from learning.behavior.behavior_engine import add_to_semantic_memory, search_semantic_memory

# Rest of the file stays exactly the same...

# Enhanced Logging
logger = logging.getLogger("jarvis")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_structured(event, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))

# Enhanced Metrics
REQUESTS = Counter('agent_requests_total', 'Total requests')
INTELLIGENT_RESPONSES = Counter('intelligent_responses_total', 'Intelligent responses generated')
LLM_CALLS = Counter('llm_calls_total', 'LLM API calls')
RESPONSE_QUALITY = Histogram('response_quality_score', 'Response quality scores')

class IntelligentJarvis:
    """Intelligent Jarvis AI Assistant"""
    
    def __init__(self):
        self.orchestrator = orchestrator
        self.specialized_agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized agents"""
        try:
            self.specialized_agents['weather'] = WeatherAgent()
            self.specialized_agents['routine'] = RoutineAgent()
            
            logger.info("✅ Intelligent agents initialized")
            logger.info(f"🧠 Available agents: {list(self.orchestrator.agent_registry._agents.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
    
    async def process_request_intelligently(self, user_request: str) -> Dict[str, Any]:
        """Process any request intelligently"""
        REQUESTS.inc()
        INTELLIGENT_RESPONSES.inc()
        start_time = time.time()
        
        try:
            # Create intelligent task
            task = AgentTask(
                task_id=str(uuid.uuid4()),
                task_type="intelligent_request",
                content=user_request,
                context={"timestamp": time.time(), "user_id": "default"},
                requester_id="intelligent_interface"
            )
            
            # Process through intelligent orchestrator
            response = await self.orchestrator.handle_task_direct(task)
            
            duration = time.time() - start_time
            RESPONSE_QUALITY.observe(response.confidence)
            
            return {
                "success": response.success,
                "response": response.result.get("workflow_result", "I'm thinking..."),
                "confidence": response.confidence,
                "reasoning": response.reasoning,
                "duration": duration,
                "metadata": response.metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Intelligent processing failed: {e}")
            
            # Fallback to basic LLM response
            from agents.core.base_agent import IntelligentLLMInterface
            llm = IntelligentLLMInterface()
            
            fallback_response = await llm.think(
                f"The user said: {user_request}. Please provide a helpful response.",
                {"fallback": True}
            )
            
            return {
                "success": True,
                "response": fallback_response,
                "confidence": 0.6,
                "reasoning": "Used fallback LLM response",
                "duration": time.time() - start_time,
                "fallback": True
            }

# Initialize intelligent Jarvis
jarvis = IntelligentJarvis()

# CLI Commands
app = typer.Typer()

@app.command()
def chat(message: str):
    """Intelligent chat with Jarvis"""
    log_structured("intelligent_chat", message=message)
    
    # Process intelligently
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(jarvis.process_request_intelligently(message))
    loop.close()
    
    # Display intelligent response
    if result["success"]:
        print(f"🧠 Jarvis: {result['response']}")
        
        confidence_emoji = "🔥" if result['confidence'] > 0.8 else "💡" if result['confidence'] > 0.6 else "🤔"
        print(f"{confidence_emoji} Confidence: {result['confidence']:.2%} | Duration: {result['duration']:.2f}s")
        
        if result.get("metadata") and result["metadata"].get("specialist_used"):
            print(f"🔧 Used specialist: {result['metadata']['specialist_used']}")
        
        if result.get("fallback"):
            print("⚡ Used fallback reasoning")
    else:
        print(f"❌ Something went wrong: {result.get('response', 'Unknown error')}")

@app.command()
def memory_search(query: str):
    """Search semantic memory intelligently"""
    results = search_semantic_memory(query, n_results=5)
    
    if results and results.get('documents'):
        print(f"🔍 Found {len(results['documents'][0])} relevant memories:")
        for i, doc in enumerate(results['documents'][0][:3]):
            print(f"   {i+1}. {doc[:100]}...")
    else:
        print("🔍 No relevant memories found")

@app.command()
def agent_status():
    """Show intelligent agent status"""
    print("🧠 Intelligent Multi-Agent System Status:")
    print(f"   🎯 Orchestrator: {jarvis.orchestrator.state.value}")
    
    for name, agent in jarvis.specialized_agents.items():
        print(f"   🔧 {name.title()}: {agent.state.value}")
        print(f"      📊 Capabilities: {len(agent.capabilities)}")
    
    # Show recent intelligent interactions
    try:
        conn = db_manager.get_connection()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM agent_logs 
                WHERE agent_id = 'orchestrator_agent' 
                AND timestamp > NOW() - INTERVAL '1 hour'
            """)
            recent_count = cur.fetchone()[0]
        
        print(f"\n🔄 Recent interactions: {recent_count} in the last hour")
    except Exception as e:
        print(f"📊 Status check error: {e}")

@app.command()
def serve():
    """Start intelligent Jarvis service"""
    # Start metrics
    start_http_server(8001)
    print("📊 Metrics server started on port 8001")
    
    # FastAPI for intelligent endpoints
    fastapi_app = FastAPI(title="Intelligent Jarvis AI", version="3.0.0")
    
    @fastapi_app.post("/intelligent/chat")
    async def intelligent_chat(request: Dict[str, Any]):
        """Intelligent chat endpoint"""
        message = request.get("message", "")
        if not message:
            return {"error": "Message required"}
        
        result = await jarvis.process_request_intelligently(message)
        return result
    
    @fastapi_app.get("/health")
    async def health():
        """Health check"""
        return {
            "status": "healthy",
            "intelligence": "active",
            "agents": len(jarvis.specialized_agents),
            "orchestrator": jarvis.orchestrator.state.value
        }
    
    # Start FastAPI
    import uvicorn
    import threading
    
    def run_api():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8005, log_level="info")
    
    threading.Thread(target=run_api, daemon=True).start()
    
    print("🚀 Intelligent Jarvis AI started!")
    print("🧠 Intelligent Chat: http://0.0.0.0:8005/intelligent/chat")
    print("❤️  Health: http://0.0.0.0:8005/health")
    
    # Intelligent service loop
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("🛑 Intelligent Jarvis shutting down...")
        running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    while running:
        try:
            # Update heartbeats
            for agent in jarvis.specialized_agents.values():
                db_manager.update_agent_heartbeat(agent.agent_id)
            
            db_manager.update_agent_heartbeat(jarvis.orchestrator.agent_id)
            
            print("🧠 Intelligent Jarvis operational")
            time.sleep(30)
        except Exception as e:
            print(f"❌ Service error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    app()
