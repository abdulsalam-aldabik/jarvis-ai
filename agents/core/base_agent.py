import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from autogen_core import RoutedAgent, MessageContext, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from dataclasses import dataclass
from config.settings import settings
from agents.core.database import db_manager

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
    """AutoGen-native base agent with A2A protocol support"""
    
    def __init__(self, agent_type: str, description: str = ""):
        super().__init__(description=description or f"{agent_type} agent using AutoGen")
        self.agent_type = agent_type
        self.agent_id = f"{agent_type}_agent"
        
        # Register in database
        db_manager.register_agent(
            agent_id=self.agent_id,
            agent_type=agent_type,
            capabilities={"description": description}
        )
        
        # Update heartbeat
        db_manager.update_agent_heartbeat(self.agent_id)
        
        # Register A2A card after initialization
        self._register_a2a_card()
    
    def _register_a2a_card(self):
        """Register this agent's A2A card"""
        try:
            from agents.core.a2a_protocol import a2a_registry, A2AAgentCard, A2ASkill
            
            card = A2AAgentCard(
                agent_id=self.agent_id,
                name=f"{self.agent_type.title()} Agent",
                description=self._get_a2a_description(),
                version="1.0.0",
                hosted_info={
                    "base_url": f"http://localhost:8005/a2a/{self.agent_id}",
                    "health_endpoint": f"http://localhost:8005/a2a/{self.agent_id}/health",
                    "discovery_endpoint": f"http://localhost:8005/.well-known/agent.json"
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
            # A2A protocol not available, continue without it
            db_manager.log_event("WARNING", f"A2A protocol not available: {e}", {}, self.agent_id)
    
    def _get_a2a_description(self) -> str:
        """Get A2A description for this agent"""
        descriptions = {
            "weather": "Provides real-time weather information and forecasts for any global location using AccuWeather API",
            "routine": "Creates, manages, and optimizes daily routines and schedules based on user preferences and learned patterns",
            "orchestrator": "Coordinates multi-agent workflows and manages task delegation across specialized agents"
        }
        return descriptions.get(self.agent_type, f"Specialized {self.agent_type} agent")
    
    def _get_a2a_skills(self) -> List:
        """Get A2A skills for this agent"""
        try:
            from agents.core.a2a_protocol import A2ASkill
            
            # Default skills based on agent type
            default_skills = {
                "weather": [
                    A2ASkill(
                        name="get_weather",
                        description="Get current weather conditions for a location",
                        input_schema={"location": "string"},
                        output_schema={"temperature": "string", "conditions": "string", "humidity": "number"},
                        examples=[{
                            "input": {"location": "Brussels"},
                            "output": {"temperature": "15°C", "conditions": "Partly cloudy", "humidity": 78}
                        }]
                    )
                ],
                "routine": [
                    A2ASkill(
                        name="create_routine",
                        description="Create a new daily routine based on user preferences",
                        input_schema={"routine_type": "string", "preferences": "object"},
                        output_schema={"routine_id": "string", "schedule": "array"},
                        examples=[{
                            "input": {"routine_type": "morning", "preferences": {"duration": 30}},
                            "output": {"routine_id": "morning_001", "schedule": ["wake_up", "exercise", "breakfast"]}
                        }]
                    )
                ],
                "orchestrator": [
                    A2ASkill(
                        name="coordinate_workflow",
                        description="Coordinate multi-agent workflows and task delegation",
                        input_schema={"task": "string", "agents_needed": "array"},
                        output_schema={"workflow_id": "string", "status": "string", "result": "object"},
                        examples=[{
                            "input": {"task": "Get weather and plan routine", "agents_needed": ["weather", "routine"]},
                            "output": {"workflow_id": "wf_001", "status": "completed", "result": {"weather": "sunny", "routine": "outdoor_activities"}}
                        }]
                    )
                ]
            }
            
            return default_skills.get(self.agent_type, [])
            
        except ImportError:
            return []
    
    async def handle_a2a_request(self, from_agent: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle A2A request from another agent"""
        try:
            # Log incoming A2A request
            db_manager.log_event(
                "INFO",
                f"A2A request from {from_agent}",
                {"task": task, "from_agent": from_agent},
                agent_id=self.agent_id
            )
            
            # Process the task
            if hasattr(self, 'process_message'):
                # Create mock context for A2A requests
                class A2AContext:
                    def __init__(self, sender):
                        self.sender = sender
                
                result = await self.process_message(task.get("content", ""), A2AContext(from_agent))
                
                response = {
                    "success": True,
                    "result": result,
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
            else:
                response = {
                    "success": False,
                    "error": "Agent does not support A2A requests",
                    "agent_id": self.agent_id,
                    "timestamp": time.time()
                }
            
            # Log the communication
            try:
                from agents.core.a2a_protocol import a2a_registry
                a2a_registry.log_communication(from_agent, self.agent_id, task, response)
            except ImportError:
                pass
            
            return response
            
        except Exception as e:
            error_response = {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            try:
                from agents.core.a2a_protocol import a2a_registry
                a2a_registry.log_communication(from_agent, self.agent_id, task, error_response)
            except ImportError:
                pass
                
            return error_response
    
    @message_handler
    async def handle_user_message(self, message: str, ctx: MessageContext) -> str:
        """AutoGen native message handler"""
        try:
            # Log the interaction
            db_manager.log_event("INFO", f"Processing: {message[:100]}", 
                               {"sender": str(ctx.sender)}, self.agent_id)
            
            # Process using agent-specific logic
            response = await self.process_message(message, ctx)
            
            # Store in semantic memory
            from learning.behavior.behavior_engine import add_to_semantic_memory
            add_to_semantic_memory(
                f"User: {message}\n{self.agent_type}: {response}",
                {"type": "conversation", "agent": self.agent_type, "timestamp": time.time()}
            )
            
            return response
            
        except Exception as e:
            db_manager.log_event("ERROR", f"Message handling failed: {str(e)}", 
                               {"message": message[:100]}, self.agent_id)
            return f"I'm having trouble processing that request: {str(e)}"
    
    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Override this in specialized agents"""
        return f"I'm a {self.agent_type} agent. I received: {message}"
