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
from src.shared_context import shared_context_manager, ConversationTurn
from src.tone_adapter import tone_adapter, ResponseContext

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
    
    async def process_message(self, message: str, session_id: str | None = None) -> str:
        """
        Process message using INTELLIGENT AutoGen 0.6.2 patterns:
        intent → enriched context → agent → LLM tone adapter → learning ack
        """
        try:
            if not self.initialized:
                self.initialize_agents()

            session_id = session_id or str(uuid.uuid4())
            shared_context = shared_context_manager.get_or_create_context(session_id)

            # tie every agent to this session
            for agent in self.agents.values():
                if hasattr(agent, "set_session_context"):
                    agent.set_session_context(session_id)

            # PHASE 1: intent
            intent, confidence, target_agent = await intent_classifier.classify_intent(message)

            # PHASE 2: enrich
            enriched_context = await context_enricher.enrich_context(
                user_message=message,
                intent=intent,
                confidence=confidence,
                target_agent=target_agent,
                session_id=session_id,
            )

            # PHASE 3: pick agent
            selected_agent = self._select_agent_with_context(enriched_context, shared_context)

            # PHASE 4: agent executes
            context_dict = self._convert_context_to_dict(enriched_context)
            if hasattr(enriched_context, "follow_up_questions"):
                context_dict["follow_up_questions"] = enriched_context.follow_up_questions

            agent_context = shared_context.get_context_for_agent(selected_agent.agent_id)
            context_dict.update(
                {
                    "shared_context": agent_context,
                    "conversation_theme": shared_context.conversation_theme,
                    "communication_style": shared_context.user_communication_style,
                    "pending_follow_ups": shared_context.pending_follow_ups.copy(),
                }
            )

            draft = await selected_agent.process_message(message, context_dict)

            # PHASE 4B: LLM tone refinement
            draft = await tone_adapter.generate_response(
                ResponseContext(
                    user_message=message,
                    agent_type=selected_agent.agent_type,
                    raw_data={"draft": draft},
                    user_preferences=shared_context.user_preferences,
                    communication_style=shared_context.user_communication_style,
                    # conversation_history=shared_context.last_messages(8),
                    # time_context=shared_context.time_of_day,
                    intent=intent,
                ),
                shared_context,
            )

            # PHASE 5: add learning acks
            enhanced_response = self._add_learning_acknowledgments(draft, shared_context)

            # update memory & logs
            shared_context.pending_follow_ups.clear()
            if hasattr(selected_agent, "store_interaction"):
                await selected_agent.store_interaction(message, enhanced_response)

            turn = ConversationTurn(
                user_message=message,
                agent_response=enhanced_response,
                agent_id=selected_agent.agent_id,
                intent=intent,
                confidence=confidence,
                timestamp=time.time(),
                context_used=context_dict,
            )
            shared_context.add_turn(turn)

            log_structured(
                "autogen_0.6.2_phase2_success",
                session_id=session_id,
                intent=intent,
                confidence=confidence,
                agent_used=selected_agent.agent_id,
                response_length=len(enhanced_response),
                conversation_turns=len(shared_context.conversation_turns),
                communication_style=shared_context.user_communication_style,
            )
            return enhanced_response

        except Exception as e:
            log_structured("autogen_0.6.2_intelligent_failed", error=str(e))
            return f"I encountered an error: {e}"

        
    def _select_agent_with_context(self, context: Any, shared_context: Any) -> AgentBase:
        """
        Intelligent agent selection based on enriched context and shared context
        """
        target_agent = context.target_agent
        
        # Primary agent selection based on intent
        if target_agent in self.agents:
            selected_agent = self.agents[target_agent]
        else:
            selected_agent = self.agents["orchestrator"]
        
        # Context-aware agent switching with shared context
        time_context = context.time_context
        user_preferences = context.user_preferences
        
        # Check if we should continue with previous agent for conversation continuity
        if len(shared_context.conversation_turns) > 0:
            last_turn = shared_context.conversation_turns[-1]
            
            # If same topic and recent interaction, prefer same agent for continuity
            if (shared_context.conversation_theme == context.intent and 
                time.time() - last_turn.timestamp < 300):  # 5 minutes
                if last_turn.agent_id in self.agents:
                    log_structured("agent_continuity_selected",
                                reason="conversation_continuity",
                                previous_agent=last_turn.agent_id,
                                current_intent=context.intent)
                    return self.agents[last_turn.agent_id]

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
            return self.agents["orchestrator"]
        
        return selected_agent
    
    def _add_learning_acknowledgments(self, response: str, shared_context: Any) -> str:
        """
        Add learning acknowledgments to the response based on shared context
        """
        try:
            # Get any pending follow-ups (learning acknowledgments)
            follow_ups = shared_context.pending_follow_ups.copy()
            
            if not follow_ups:
                return response
            
            # Add learning acknowledgments to the response
            enhanced_response = response
            
            # Add a natural transition
            if len(follow_ups) == 1:
                enhanced_response += f"\n\n💡 {follow_ups[0]}"
            else:
                enhanced_response += f"\n\n💡 A few things I've learned:\n"
                for i, acknowledgment in enumerate(follow_ups, 1):
                    enhanced_response += f"• {acknowledgment}\n"
            
            log_structured("learning_acknowledgments_added",
                         response_length=len(enhanced_response),
                         acknowledgments_count=len(follow_ups),
                         session_id=shared_context.session_id[:8])
            
            return enhanced_response
            
        except Exception as e:
            log_structured("learning_acknowledgments_failed", error=str(e))
            return response

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
                "system_type": "AutoGen 0.6.2 Intelligent GroupChatManager Phase 2",
                "autogen_version": "0.6.2-official",
                "intelligence_features": [
                    "intent_classification",
                    "context_enrichment", 
                    "adaptive_agent_selection",
                    "preference_learning",
                    "shared_context",
                    "learning_acknowledgments"
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
