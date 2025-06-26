"""
Agentic state management for reasoning cycles
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json
import uuid


class ReasoningStep(str, Enum):
    """Reasoning cycle steps"""
    REASON = "reason"
    PLAN = "plan" 
    ACT = "act"
    OBSERVE = "observe"
    ADAPT = "adapt"


class AgentAction(BaseModel):
    """Represents an action taken by an agent"""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expected_outcome: Optional[str] = None


class AgentObservation(BaseModel):
    """Represents an observation from an action"""
    observation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_id: str
    result: Any
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReasoningState(BaseModel):
    """State for agent reasoning cycle"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    current_step: ReasoningStep = ReasoningStep.REASON
    user_input: str = ""
    
    # Reasoning cycle data
    reasoning: Optional[str] = None
    plan: List[AgentAction] = Field(default_factory=list)
    current_action_index: int = 0
    observations: List[AgentObservation] = Field(default_factory=list)
    adaptations: List[str] = Field(default_factory=list)
    
    # Context and memory
    context: Dict[str, Any] = Field(default_factory=dict)
    memory_retrieved: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Final response
    final_response: Optional[str] = None
    completed: bool = False
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def update_step(self, step: ReasoningStep):
        """Update current reasoning step"""
        self.current_step = step
        self.updated_at = datetime.utcnow()
    
    def add_observation(self, action_id: str, result: Any, success: bool, error_message: Optional[str] = None):
        """Add observation from action execution"""
        observation = AgentObservation(
            action_id=action_id,
            result=result,
            success=success,
            error_message=error_message
        )
        self.observations.append(observation)
        self.updated_at = datetime.utcnow()
        return observation
    
    def get_current_action(self) -> Optional[AgentAction]:
        """Get current action to execute"""
        if self.current_action_index < len(self.plan):
            return self.plan[self.current_action_index]
        return None
    
    def advance_action(self):
        """Move to next action in plan"""
        self.current_action_index += 1
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return json.loads(self.model_dump_json())
