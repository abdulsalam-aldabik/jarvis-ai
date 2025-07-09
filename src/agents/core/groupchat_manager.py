# src/agents/core/groupchat_manager.py (Updated)
"""
AutoGen 0.6.2 GroupChatManager with Intent-Driven Routing
Replaces keyword matching with intelligent intent classification
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient

from src.agents.core.database import db_manager

from src.agents.core.base_agent import AgentBase, get_model_client
from src.agents.core.logging_config import log_structured
from src.intent_classifier import intent_classifier
from src.context_enricher import context_enricher
from config.settings import settings

class GroupChatManager:
    """
    INTELLIGENT AutoGen 0.6.2 GroupChatManager with Intent-Driven Routing
    Replaces keyword matching with semantic intent classification
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.group_chat = None
        self.initialized = False
        self.database = db_manager

    def initialize_agents(self):
        """Initialize specialized AssistantAgent personas"""
        try:
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
            log_structured("autogen_0.6.2_intelligent_groupchat_initialized",
                         agents_count=len(self.agents),
                         autogen_version="0.6.2-official",
                         intelligence_mode="intent_driven")
            
        except Exception as e:
            log_structured("groupchat_initialization_failed", error=str(e))
            raise
    
    async def process_message(self, message: str, session_id: str = None) -> str:
        """
        Process message using INTELLIGENT AutoGen 0.6.2 patterns
        Replaces keyword matching with intent classification and context enrichment
        """
        try:
            if not self.initialized:
                self.initialize_agents()
                
            session_id = session_id or str(uuid.uuid4())
            
            # Set session context for all agents
            for agent in self.agents.values():
                if hasattr(agent, 'set_session_context'):
                    agent.set_session_context(session_id)
            
            # PHASE 1: Intent Classification
            intent, confidence, target_agent = await intent_classifier.classify_intent(message)
            
            # PHASE 2: Context Enrichment
            enriched_context = await context_enricher.enrich_context(
                user_message=message,
                intent=intent,
                confidence=confidence,
                target_agent=target_agent,
                session_id=session_id
            )
            
            # PHASE 3: Intelligent Agent Selection
            selected_agent = self._select_agent_with_context(enriched_context)
            
            # PHASE 4: Context-Aware Processing
            context_dict = self._convert_context_to_dict(enriched_context)
            response = await selected_agent.process_message(message, context_dict)

        
            
            log_structured("autogen_0.6.2_intelligent_success",
                         session_id=session_id,
                         intent=intent,
                         confidence=confidence,
                         agent_used=selected_agent.agent_id,
                         response_length=len(response))
            
            return response
            
        except Exception as e:
            log_structured("autogen_0.6.2_intelligent_failed", error=str(e))
            return f"I encountered an error: {str(e)}"
    
    def _select_agent_with_context(self, context: Any) -> AgentBase:
        """
        Intelligent agent selection based on enriched context
        Replaces simple keyword matching with contextual decision making
        """
        target_agent = context.target_agent
        
        # Primary agent selection based on intent
        if target_agent in self.agents:
            selected_agent = self.agents[target_agent]
        else:
            selected_agent = self.agents["orchestrator"]
        
        # Context-aware agent switching
        time_context = context.time_context
        user_preferences = context.user_preferences
        
        # Morning routine optimization
        if (context.intent == "routine_create" and 
            time_context.get("is_morning", False) and
            "morning" in user_preferences.get("preferred_routine_types", [])):
            log_structured("context_aware_agent_selection",
                         reason="morning_routine_optimization",
                         selected_agent="routine")
            return self.agents["routine"]
        
        # Weather-dependent routine suggestions
        if (context.intent == "routine_create" and 
            any(word in context.user_message.lower() for word in ["outside", "outdoor", "weather"])):
            log_structured("context_aware_agent_selection",
                         reason="weather_dependent_routing",
                         selected_agent="orchestrator")
            return self.agents["orchestrator"]  # Orchestrator will coordinate weather+routine
        
        return selected_agent
    
    def _convert_context_to_dict(self, context: Any) -> Dict[str, Any]:
        """Convert enriched context to dictionary for agent processing"""
        return {
            "intent": context.intent,
            "confidence": context.confidence,
            "session_id": context.session_id,
            "user_preferences": context.user_preferences,
            "conversation_history": context.conversation_history,
            "time_context": context.time_context,
            "system_context": context.system_context,
            "agent_memory": context.agent_memory,
            "previous_interactions": context.previous_interactions,
            "intelligence_mode": "context_aware"
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive intelligent GroupChat system status"""
        try:
            return {
                "status": "healthy",
                "system_type": "AutoGen 0.6.2 Intelligent GroupChatManager",
                "autogen_version": "0.6.2-official",
                "intelligence_features": [
                    "intent_classification",
                    "context_enrichment", 
                    "adaptive_agent_selection",
                    "preference_learning"
                ],
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
    
    async def run_stream(self, task: str, session_id: str = None):
        """
        Stream-based message processing with intelligent features
        """
        try:
            if not self.initialized:
                self.initialize_agents()
                
            session_id = session_id or str(uuid.uuid4())
            
            # Set session context for all agents
            for agent in self.agents.values():
                if hasattr(agent, 'set_session_context'):
                    agent.set_session_context(session_id)
            
            # Process with intelligence pipeline
            response = await self.process_message(task, session_id)
            
            # Stream the response
            async def stream_generator():
                words = response.split()
                for i, word in enumerate(words):
                    yield f"{word} "
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            return stream_generator()
            
        except Exception as e:
            log_structured("groupchat_intelligent_stream_failed", error=str(e))
            
            async def error_generator():
                yield f"I encountered an error: {str(e)}"
            
            return error_generator()

# Global GroupChat manager instance
groupchat_manager = GroupChatManager()
