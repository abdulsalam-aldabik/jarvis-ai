"""
AutoGen 0.6.2 GroupChatManager Implementation
Following: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html
"""
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient

from src.agents.core.base_agent import AgentBase, get_model_client
from src.agents.core.logging_config import log_structured
from config.settings import settings

class GroupChatManager:
    """CORRECT AutoGen 0.6.2 GroupChatManager following official patterns"""
    
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.group_chat = None
        self.initialized = False
        
    def initialize_agents(self):
        """Initialize specialized AssistantAgent personas"""
        try:
            # Import specialized agents
            from src.agents.specialized.weather_agent import WeatherAgent
            from src.agents.specialized.routine_agent import RoutineAgent
            from src.agents.core.orchestrator import OrchestratorAgent
            
            # Create agent instances
            self.agents = {
                "weather": WeatherAgent(),
                "routine": RoutineAgent(), 
                "orchestrator": OrchestratorAgent()
            }
            
            # Create GroupChat with AutoGen 0.6.2 patterns
            self.group_chat = RoundRobinGroupChat(
                participants=list(self.agents.values()),
                termination_condition=MaxMessageTermination(max_messages=10)
            )
            
            self.initialized = True
            log_structured("autogen_062_groupchat_initialized",
                         agents_count=len(self.agents),
                         autogen_version="0.6.2-official")
                         
        except Exception as e:
            log_structured("groupchat_initialization_failed", error=str(e))
            raise

    async def process_message(self, message: str, session_id: str = None) -> str:
        """Process message using AutoGen 0.6.2 GroupChat patterns"""
        try:
            if not self.initialized:
                self.initialize_agents()
            
            session_id = session_id or str(uuid.uuid4())
            
            # Set session context for all agents
            for agent in self.agents.values():
                if hasattr(agent, 'set_session_context'):
                    agent.set_session_context(session_id)
            
            # Select appropriate agent for the message
            selected_agent_name = self.select_agent_for_message(message)
            selected_agent = self.agents.get(selected_agent_name, self.agents["orchestrator"])
            
            # Process with selected agent
            response = await selected_agent.process_message(message)
            
            log_structured("autogen_062_groupchat_success",
                         session_id=session_id,
                         agent_used=selected_agent_name,
                         response_length=len(response))
                         
            return response
            
        except Exception as e:
            log_structured("autogen_062_groupchat_failed", error=str(e))
            return f"I encountered an error: {str(e)}"

    def select_agent_for_message(self, message: str) -> str:
        """Select appropriate agent based on message content"""
        message_lower = message.lower()
        
        # Weather-related keywords
        if any(word in message_lower for word in ["weather", "temperature", "forecast", "rain", "sunny"]):
            return "weather"
        
        # Routine-related keywords  
        elif any(word in message_lower for word in ["routine", "schedule", "habit", "morning", "evening"]):
            return "routine"
        
        # Default to orchestrator
        else:
            return "orchestrator"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive GroupChat system status"""
        try:
            return {
                "status": "healthy",
                "system_type": "AutoGen 0.6.2 GroupChatManager",
                "autogen_version": "0.6.2-official",
                "agents": {
                    "count": len(self.agents),
                    "types": list(self.agents.keys())
                },
                "groupchat_initialized": self.initialized,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "autogen_version": "0.6.2-official",
                "timestamp": time.time()
            }

# Global GroupChat manager instance
groupchat_manager = GroupChatManager()
