"""
Simplified agentic state for LangGraph TypedDict compatibility
"""
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
from enum import Enum
import json
import uuid
from operator import add

class ReasoningStep(str, Enum):
    """Simplified reasoning steps for LangGraph"""
    ANALYZE = "analyze"
    PLAN = "plan" 
    EXECUTE = "execute"
    FINALIZE = "finalize"

class SimpleAction(TypedDict):
    """Simplified action structure compatible with LangGraph"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: float

class SimpleObservation(TypedDict):
    """Simplified observation structure compatible with LangGraph"""
    observation_id: str
    action_id: str
    result: Any
    success: bool
    error_message: Optional[str]
    timestamp: float

# This is the MAIN state that LangGraph will use - defined in agentic_workflow.py
# We keep this file for backward compatibility but the real state is HybridAgentState
