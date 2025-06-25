import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException
from config.settings import settings
from agents.core.database import db_manager

@dataclass
class A2ASkill:
    """A2A skill definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]] = None

@dataclass
class A2AAgentCard:
    """A2A Agent Card following Google's A2A protocol"""
    agent_id: str
    name: str
    description: str
    version: str
    hosted_info: Dict[str, str]
    skills: List[A2ASkill]
    communication_methods: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2A protocol format"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "hosted_info": self.hosted_info,
            "skills": [asdict(skill) for skill in self.skills],
            "communication_methods": self.communication_methods,
            "metadata": self.metadata or {},
            "last_updated": time.time()
        }

class A2ARegistry:
    """A2A Agent Registry for discovery and communication"""
    
    def __init__(self):
        self.agents: Dict[str, A2AAgentCard] = {}
        self.communication_log: List[Dict[str, Any]] = []
    
    def register_agent(self, agent_card: A2AAgentCard):
        """Register an agent in the A2A registry"""
        self.agents[agent_card.agent_id] = agent_card
        
        # Log registration
        db_manager.log_event(
            "INFO",
            f"A2A agent registered: {agent_card.name}",
            {
                "agent_id": agent_card.agent_id,
                "skills_count": len(agent_card.skills),
                "version": agent_card.version
            },
            agent_id=agent_card.agent_id
        )
    
    def discover_agents(self, skill_filter: Optional[str] = None) -> List[A2AAgentCard]:
        """Discover available agents, optionally filtered by skill"""
        if not skill_filter:
            return list(self.agents.values())
        
        matching_agents = []
        for agent in self.agents.values():
            for skill in agent.skills:
                if skill_filter.lower() in skill.name.lower() or skill_filter.lower() in skill.description.lower():
                    matching_agents.append(agent)
                    break
        
        return matching_agents
    
    def get_agent_card(self, agent_id: str) -> Optional[A2AAgentCard]:
        """Get specific agent card"""
        return self.agents.get(agent_id)
    
    def log_communication(self, from_agent: str, to_agent: str, task: Dict[str, Any], result: Dict[str, Any]):
        """Log A2A communication"""
        comm_log = {
            "timestamp": time.time(),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "task": task,
            "result": result,
            "success": result.get("success", False)
        }
        
        self.communication_log.append(comm_log)
        
        # Store in database
        db_manager.store_agent_interaction(
            agent_id=from_agent,
            interaction_type="a2a_communication",
            data=comm_log
        )

# Global A2A registry
a2a_registry = A2ARegistry()

class A2AProtocolMixin:
    """Mixin to add A2A protocol support to agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_a2a_card()
    
    def _register_a2a_card(self):
        """Register this agent's A2A card"""
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
    
    def _get_a2a_description(self) -> str:
        """Get A2A description for this agent"""
        descriptions = {
            "weather": "Provides real-time weather information and forecasts for any global location using AccuWeather API",
            "routine": "Creates, manages, and optimizes daily routines and schedules based on user preferences and learned patterns",
            "orchestrator": "Coordinates multi-agent workflows and manages task delegation across specialized agents"
        }
        return descriptions.get(self.agent_type, f"Specialized {self.agent_type} agent")
    
    def _get_a2a_skills(self) -> List[A2ASkill]:
        """Get A2A skills for this agent"""
        if hasattr(self, 'capabilities'):
            return [
                A2ASkill(
                    name=cap.name,
                    description=cap.description,
                    input_schema=cap.input_schema,
                    output_schema=cap.output_schema,
                    examples=[{
                        "input": {"example": "sample input"},
                        "output": {"example": "sample output"}
                    }]
                )
                for cap in self.capabilities
            ]
        
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
            ]
        }
        
        return default_skills.get(self.agent_type, [])
    
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
            a2a_registry.log_communication(from_agent, self.agent_id, task, response)
            
            return response
            
        except Exception as e:
            error_response = {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
            
            a2a_registry.log_communication(from_agent, self.agent_id, task, error_response)
            return error_response
