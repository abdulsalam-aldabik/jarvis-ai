import json
import time
import uuid
import asyncio
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from config.settings import settings
from agents.core.database import db_manager

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COLLABORATING = "collaborating"
    ERROR = "error"

@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

@dataclass
class AgentTask:
    task_id: str
    task_type: str
    content: str
    context: Dict[str, Any]
    requester_id: str
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentResponse:
    task_id: str
    agent_id: str
    success: bool
    result: Any
    reasoning: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class AgentRegistry:
    def __init__(self):
        self._agents = {}
    
    def register_agent(self, agent):
        self._agents[agent.agent_id] = agent
        db_manager.log_event("INFO", f"Agent registered: {agent.agent_id}", 
                            {"agent_type": agent.agent_type}, agent_id=agent.agent_id)
    
    def get_agent(self, agent_id: str):
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

agent_registry = AgentRegistry()

class IntelligentLLMInterface:
    """Smart LLM interface for agent reasoning"""
    
    @staticmethod
    async def think(prompt: str, context: Dict[str, Any] = None, agent_id: str = "system") -> str:
        """Use LLM for intelligent reasoning"""
        try:
            # Enhanced prompt with context
            system_prompt = """You are an intelligent AI agent assistant. You think step by step, provide reasoning, and give helpful responses. 
            You have access to specialized tools and other agents for specific tasks. Be concise but intelligent. Be as short and to the point and no extra information."""
            
            if context:
                context_str = f"\nContext: {json.dumps(context, indent=2)}"
                prompt = prompt + context_str
            
            payload = {
                "model": settings.llm.default_model,
                "prompt": f"System: {system_prompt}\n\nUser/Agent: {prompt}\n\nResponse:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json=payload,
                timeout=settings.llm.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"LLM Error: HTTP {response.status_code}"
                
        except Exception as e:
            db_manager.log_event("ERROR", f"LLM call failed: {str(e)}", {"agent_id": agent_id})
            return f"I'm having trouble thinking right now: {str(e)}"

class BaseAgent(ABC):
    """Intelligent base class for all agents"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        self.context = {}
        self.last_activity = time.time()
        self.llm = IntelligentLLMInterface()
        
        # Register agent
        self._register_agent()
        agent_registry.register_agent(self)
    
    def _register_agent(self):
        capabilities_dict = {
            cap.name: {
                "description": cap.description,
                "input_schema": cap.input_schema,
                "output_schema": cap.output_schema
            }
            for cap in self.capabilities
        }
        
        db_manager.register_agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            capabilities=capabilities_dict
        )
    
    def update_state(self, new_state: AgentState, context: Dict[str, Any] = None):
        old_state = self.state
        self.state = new_state
        self.last_activity = time.time()
        
        if context:
            self.context.update(context)
        
        db_manager.log_event(
            "INFO", 
            f"Agent state: {old_state.value} → {new_state.value}",
            {"context": context},
            agent_id=self.agent_id
        )
        
        db_manager.update_agent_heartbeat(self.agent_id)
    
    async def think_about_task(self, task: AgentTask) -> str:
        """Use LLM to think about the task intelligently"""
        self.update_state(AgentState.THINKING)
        
        think_prompt = f"""
        I am the {self.agent_type} agent. I need to handle this task:
        
        Task Type: {task.task_type}
        User Request: {task.content}
        Context: {task.context}
        
        My capabilities: {[cap.name for cap in self.capabilities]}
        
        How should I approach this task? What's my reasoning and plan?
        """
        
        reasoning = await self.llm.think(think_prompt, {"agent_type": self.agent_type}, self.agent_id)
        
        db_manager.log_event("INFO", f"Agent reasoning: {reasoning}", 
                            {"task_id": task.task_id}, agent_id=self.agent_id)
        
        return reasoning
    
    async def handle_task_direct(self, task: AgentTask) -> AgentResponse:
        """Intelligently handle any task"""
        try:
            # Think about the task
            reasoning = await self.think_about_task(task)
            
            # Execute the task
            self.update_state(AgentState.ACTING)
            result = await self.execute_intelligent_plan(task, reasoning)
            
            # Determine success and confidence
            success = result is not None and "error" not in str(result).lower()
            confidence = 0.85 if success else 0.3
            
            response = AgentResponse(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=success,
                result=result,
                reasoning=reasoning,
                confidence=confidence,
                metadata={"task_type": task.task_type}
            )
            
            self.update_state(AgentState.IDLE)
            return response
            
        except Exception as e:
            self.update_state(AgentState.ERROR)
            db_manager.log_event("ERROR", f"Task failed: {str(e)}", 
                                {"task_id": task.task_id}, agent_id=self.agent_id)
            
            return AgentResponse(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                result=None,
                reasoning=f"Task failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    @abstractmethod
    async def execute_intelligent_plan(self, task: AgentTask, reasoning: str) -> Any:
        """Execute the plan with intelligence - implemented by specialized agents"""
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "capabilities": [cap.name for cap in self.capabilities],
            "last_activity": self.last_activity,
        }
