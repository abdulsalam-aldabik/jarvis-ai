"""
Enhanced base agent with proper AutoGen 2.x patterns and LangGraph integration
"""
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from autogen_core import RoutedAgent, MessageContext, message_handler
from dataclasses import dataclass
from config.settings import settings
from src.agents.core.database import db_manager
from datetime import datetime
from src.agents.core.logging_config import log_structured

@dataclass
class AgentTask:
    """Simple task structure for AutoGen messaging"""
    content: str
    context: Dict[str, Any]
    sender: str

@dataclass 
class AgentResponse:
    """Simple response structure"""
    content: str
    success: bool
    metadata: Dict[str, Any]

class AutoGenBaseAgent(RoutedAgent):
    """Enhanced base agent with proper AutoGen 2.x patterns"""
    
    def __init__(self, name: str, description: str, agent_type: str = None):
        super().__init__(name)
        self.description = description
        self.agent_id = name
        self.agent_type = agent_type or self.__class__.__name__.lower()
        self.database = db_manager  # Always use singleton
        self.current_session_id = None
        
        # Register in global registry
        agent_registry.register_agent(self)
        self._register_a2a_card()

    def set_session_context(self, session_id: str):
        """Set session context for memory operations"""
        self.current_session_id = session_id
        log_structured("agent_session_context_set", agent_id=self.agent_id, session_id=session_id)

    async def agentic_process(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process user input using enhanced reasoning"""
        try:
            # Use orchestrator for complex processing
            from src.agents.core.orchestrator import orchestrator
            
            # Set session context if available
            if self.current_session_id and hasattr(orchestrator, 'set_session_context'):
                orchestrator.set_session_context(self.current_session_id)
            
            # Create mock context for processing
            class MockContext:
                def __init__(self, sender):
                    self.sender = sender
            
            result = await orchestrator.process_message(user_input, MockContext(self.agent_id))
            await self._store_interaction(user_input, result)
            return result or "I couldn't process your request."
            
        except Exception as e:
            log_structured("agentic_process_failed", agent_id=self.agent_id, error=str(e))
            return f"I'm having trouble processing that request: {str(e)}"

    async def _store_interaction(self, user_input: str, response: str):
        """Store interaction in memory"""
        try:
            interaction_id = self.database.store_agent_interaction(
                agent_id=self.agent_id,
                interaction_type="chat",
                data={
                    "user_input": user_input,
                    "agent_response": response,
                    "metadata": {"timestamp": time.time()}
                }
            )
            if interaction_id:
                log_structured("interaction_stored", interaction_id=interaction_id)
        except Exception as e:
            log_structured("interaction_storage_failed", error=str(e))

    def _register_a2a_card(self):
        """Register this agent's A2A card"""
        try:
            from src.agents.core.a2a_protocol import a2a_registry, A2AAgentCard, A2ASkill
            
            card = A2AAgentCard(
                agent_id=self.agent_id,
                name=f"{self.agent_type.title()} Agent",
                description=self._get_a2a_description(),
                version="1.0.0",
                hosted_info={
                    "base_url": f"http://localhost:8005/a2a/{self.agent_id}",
                    "health_endpoint": f"http://localhost:8005/a2a/{self.agent_id}/health",
                    "discovery_endpoint": "http://localhost:8005/.well-known/agent.json"
                },
                skills=self._get_a2a_skills(),
                communication_methods=["request_response", "sse", "push_notification"],
                metadata={
                    "framework": "AutoGen",
                    "language": "Python", 
                    "created_at": time.time()
                }
            )
            
            a2a_registry.register_agent(card)
            
        except ImportError as e:
            self.database.log_event("WARNING", f"A2A protocol not available: {e}", "", self.agent_id)

    def _get_a2a_description(self) -> str:
        """Get A2A description for this agent"""
        descriptions = {
            "weather": "Provides real-time weather information and forecasts",
            "routine": "Creates and manages daily routines and schedules", 
            "orchestrator": "Coordinates multi-agent workflows and task delegation"
        }
        return descriptions.get(self.agent_type, f"Specialized {self.agent_type} agent")

    def _get_a2a_skills(self) -> List:
        """Get A2A skills for this agent"""
        try:
            from src.agents.core.a2a_protocol import A2ASkill
            
            default_skills = {
                "weather": [A2ASkill(
                    name="get_weather",
                    description="Get current weather conditions for a location",
                    input_schema={"location": "string"},
                    output_schema={"temperature": "string", "conditions": "string"},
                    examples=[{"input": {"location": "Brussels"}, "output": {"temperature": "15°C", "conditions": "Partly cloudy"}}]
                )]
            }
            return default_skills.get(self.agent_type, [])
            
        except ImportError:
            return []

    @message_handler
    async def handle_message(self, message: str, ctx: MessageContext) -> str:
        """AutoGen message handler"""
        try:
            self.database.log_event("INFO", f"Processing message: {message[:100]}", sender=str(ctx.sender), agent_id=self.agent_id)
            response = await self.agentic_process(message)
            return response
        except Exception as e:
            self.database.log_event("ERROR", f"Message handling failed: {str(e)}", message=message[:100], agent_id=self.agent_id)
            return f"I'm having trouble processing that request: {str(e)}"

    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process message - to be overridden by specialized agents"""
        return f"I'm a {self.agent_type} agent. I received: {message}"

class AgentRegistry:
    """Global agent registry for AutoGen agents"""
    
    def __init__(self):
        self.agents: Dict[str, AutoGenBaseAgent] = {}

    def register_agent(self, agent):
        """Register an agent in the registry"""
        try:
            agent_key = getattr(agent, 'agent_id', None) or getattr(agent, 'name', 'unknown')
            agent_type = getattr(agent, 'agent_type', 'unknown')
            
            self.agents[agent_key] = agent
            log_structured("agent_registered", agent_name=agent_key, agent_type=agent_type, agent_class=type(agent).__name__)
            
            # Register in database if agent has database connection
            if hasattr(agent, 'database') and agent.database:
                try:
                    agent.database.register_agent(
                        agent_id=agent_key,
                        agent_type=agent_type,
                        capabilities={"type": agent_type},
                        description=getattr(agent, 'description', '')
                    )
                except Exception as db_error:
                    log_structured("database_registration_failed", agent_id=agent_key, error=str(db_error))
            
            return True
            
        except Exception as e:
            log_structured("agent_registration_failed", agent_name=getattr(agent, 'name', 'unknown'), error=str(e))
            return False

    def get_agent(self, agent_id: str):
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """List all registered agent IDs"""
        return list(self.agents.keys())

# Global registry instance
agent_registry = AgentRegistry()
