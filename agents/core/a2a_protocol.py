"""
A2A (Agent-to-Agent) Protocol implementation for AutoGen agents
"""
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from agents.core.logging_config import log_structured

@dataclass
class A2ASkill:
    """A2A skill definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]]

@dataclass
class A2AAgentCard:
    """A2A agent card following protocol standards"""
    agent_id: str
    name: str
    description: str
    version: str
    hosted_info: Dict[str, str]
    skills: List[A2ASkill]
    communication_methods: List[str]
    metadata: Dict[str, Any]
    last_updated: float = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class A2ARegistry:
    """Registry for A2A agent cards"""
    
    def __init__(self):
        self.agents: Dict[str, A2AAgentCard] = {}
    
    def register_agent(self, card: A2AAgentCard):
        """Register an A2A agent card"""
        self.agents[card.agent_id] = card
        log_structured("a2a_agent_registered", agent_id=card.agent_id, name=card.name)
    
    def get_agent_card(self, agent_id: str) -> Optional[A2AAgentCard]:
        """Get agent card by ID"""
        return self.agents.get(agent_id)
    
    def discover_agents(self) -> List[A2AAgentCard]:
        """Discover all registered agents"""
        return list(self.agents.values())

# Global A2A registry
a2a_registry = A2ARegistry()
