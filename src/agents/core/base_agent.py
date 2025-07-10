"""
AutoGen 0.6.2 AssistantAgent Base - Enhanced Implementation with Dynamic LLM Integration
Following: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
"""
import asyncio
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# CORRECT AutoGen 0.6.2 AgentChat imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

from config.settings import settings
from src.agents.core.database import db_manager
from src.agents.core.logging_config import log_structured
from src.tone_adapter import tone_adapter, ResponseContext


class EmotionalTone(Enum):
    """Emotional tone indicators for dynamic response adaptation"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CONCERNED = "concerned"
    EXCITED = "excited"
    HELPFUL = "helpful"
    EMPATHETIC = "empathetic"
    PROFESSIONAL = "professional"
    CASUAL = "casual"


@dataclass
class AgentMessage:
    """Enhanced message structure for agent communication"""
    content: str
    sender: str
    timestamp: float = None
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    intent: str = "general"
    confidence: float = 0.0
    context_used: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class LearningInsight:
    """Structure for capturing learning insights"""
    insight_type: str
    content: str
    confidence: float
    timestamp: float
    source: str
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp'):
            self.timestamp = time.time()


@dataclass
class ProactiveSuggestion:
    """Structure for proactive suggestions"""
    suggestion: str
    context: str
    priority: int
    expires_at: float
    suggestion_type: str
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class PatternMiner:
    """Enhanced pattern mining for predictive intelligence"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.interaction_patterns: List[Dict[str, Any]] = []
        self.learned_preferences: Dict[str, Any] = {}
        self.behavioral_patterns: Dict[str, float] = {}
    
    async def analyze_interaction(self, user_input: str, context: Dict[str, Any]) -> List[LearningInsight]:
        """Analyze interaction for learning patterns"""
        insights = []
        
        # Time-based pattern analysis
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10:
            time_pattern = "morning_activity"
        elif 11 <= current_hour <= 14:
            time_pattern = "midday_activity"
        elif 15 <= current_hour <= 18:
            time_pattern = "afternoon_activity"
        else:
            time_pattern = "evening_activity"
        
        # Communication style learning
        if any(word in user_input.lower() for word in ["please", "thank you", "could you"]):
            insights.append(LearningInsight(
                insight_type="communication_style",
                content="User prefers polite, formal communication",
                confidence=0.8,
                timestamp=time.time(),
                source="pattern_analysis"
            ))
        
        # Task preference learning
        if "weather" in user_input.lower() and time_pattern == "morning_activity":
            insights.append(LearningInsight(
                insight_type="routine_preference",
                content="User frequently checks weather in the morning",
                confidence=0.9,
                timestamp=time.time(),
                source="temporal_pattern"
            ))
        
        # Emotional state detection
        if any(word in user_input.lower() for word in ["urgent", "quickly", "asap"]):
            insights.append(LearningInsight(
                insight_type="emotional_state",
                content="User appears to be in a hurry",
                confidence=0.7,
                timestamp=time.time(),
                source="urgency_detection"
            ))
        
        return insights
    
    async def generate_proactive_suggestions(self, context: Dict[str, Any]) -> List[ProactiveSuggestion]:
        """Generate proactive suggestions based on learned patterns"""
        suggestions = []
        current_time = time.time()
        
        # Morning routine suggestions
        if datetime.now().hour == 7:
            suggestions.append(ProactiveSuggestion(
                suggestion="Good morning! Would you like me to check the weather and suggest a morning routine?",
                context="morning_routine_optimization",
                priority=1,
                expires_at=current_time + 3600,  # 1 hour
                suggestion_type="routine"
            ))
        
        # Evening planning suggestions
        if datetime.now().hour == 20:
            suggestions.append(ProactiveSuggestion(
                suggestion="Planning for tomorrow? I can help you prepare based on weather and your preferences.",
                context="evening_planning",
                priority=2,
                expires_at=current_time + 7200,  # 2 hours
                suggestion_type="planning"
            ))
        
        return suggestions


def get_model_client() -> OllamaChatCompletionClient:
    """Get Ollama model client following official AutoGen 0.6.2 patterns"""
    return OllamaChatCompletionClient(
        model=settings.llm.default_model,
        base_url=f"{settings.llm.ollama_url}",
    )


class AgentBase(AssistantAgent):
    """Enhanced AutoGen 0.6.2 AssistantAgent with Full Dynamic LLM Integration"""
    
    def __init__(self, name: str, description: str, agent_type: str = None, system_message: str = None):
        # Create memory instance with a proper name
        list_memory = ListMemory(name=f"{name}_short_term")
        
        # Enhanced system message for dynamic responses
        enhanced_system_message = system_message or f"""You are {name}, a {description}. 

        Your core capabilities:
        - Generate dynamic, contextual responses that adapt to user preferences
        - Learn from interactions and adjust your communication style
        - Provide proactive assistance and suggestions
        - Maintain conversation continuity and emotional intelligence
        - Reference past interactions when relevant
        - Ask intelligent follow-up questions to better assist users
        
        Response Guidelines:
        - Be conversational and natural, not robotic
        - Adapt your tone based on user's communication style
        - Reference user preferences and past interactions when appropriate
        - Provide actionable suggestions and proactive assistance
        - Show empathy and emotional intelligence
        - Always aim to be helpful and add value to the conversation
        """
        
        # ✅ FIXED: AutoGen 0.6.2 AssistantAgent initialization with LIST of memory objects
        super().__init__(
            name=name,
            description=description,
            model_client=get_model_client(),
            system_message=enhanced_system_message,
            memory=[list_memory]  # ✅ CRITICAL FIX: List of Memory objects, not single object
        )
        
        self.agent_id = name
        self.agent_type = agent_type or self.__class__.__name__.lower()
        self.database = db_manager
        self.current_session_id = None
        
        # ✅ NEW: Enhanced intelligence components
        self.pattern_miner = PatternMiner(self.agent_id)
        self.learned_insights: List[LearningInsight] = []
        self.proactive_suggestions: List[ProactiveSuggestion] = []
        self.conversation_context: Dict[str, Any] = {}
        self.emotional_state_tracker: Dict[str, Any] = {}
        
        # Initialize vector memory with enhanced error handling
        try:
            self.vector_memory = ChromaDBVectorMemory(
                config=PersistentChromaDBVectorMemoryConfig(
                    collection_name=f"agent_{self.agent_id}_memory",
                    persistence_path=f"./memory_bank/{self.agent_id}",
                    k=5,  # Increased for better context
                    score_threshold=0.3,  # Lower threshold for more context
                    embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                        model_name="all-MiniLM-L6-v2"
                    ),
                )
            )
            log_structured("enhanced_vector_memory_initialized", 
                         agent_id=self.agent_id, 
                         intelligence_features=["pattern_mining", "proactive_suggestions", "emotional_tracking"])
        except Exception as e:
            log_structured("vector_memory_init_failed", agent_id=self.agent_id, error=str(e))
            self.vector_memory = None
            print(f"⚠️  Warning: Vector memory failed for {self.agent_id}, using basic memory only")
        
        # Register agent with enhanced capabilities
        agentregistry.register_agent(self)
        log_structured("enhanced_autogen_062_agent_initialized", 
                     agent_id=self.agent_id, 
                     agent_type=self.agent_type,
                     framework="autogen-0.6.2-official",
                     intelligence_features=["dynamic_responses", "pattern_learning", "proactive_assistance", "emotional_intelligence"],
                     memory_count=len(self._memory))

    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Enhanced message processing with full dynamic intelligence"""
        try:
            # Initialize context
            context = context or {}
            
            # ✅ PHASE 1: Learn from interaction
            insights = await self.pattern_miner.analyze_interaction(message, context)
            self.learned_insights.extend(insights)
            
            # ✅ PHASE 2: Update conversation context
            self.conversation_context.update({
                'last_user_message': message,
                'timestamp': time.time(),
                'insights_learned': len(insights),
                'emotional_indicators': self._detect_emotional_indicators(message)
            })
            
            # ✅ PHASE 3: Generate proactive suggestions
            suggestions = await self.pattern_miner.generate_proactive_suggestions(context)
            self.proactive_suggestions.extend(suggestions)
            
            # ✅ PHASE 4: Enrich context with learned information
            enriched_context = await self._enrich_context_with_intelligence(context)
            
            # Create TextMessage following AutoGen 0.6.2 patterns
            text_message = TextMessage(content=message, source="user")
            
            # Process with AssistantAgent to get base response
            result = await self.on_messages([text_message], enriched_context)
            base_response = result.messages[-1].content
            
            # ✅ PHASE 5: Apply dynamic LLM enhancement
            enhanced_response = await self.enhance_response_with_full_intelligence(
                base_response, message, enriched_context
            )
            
            # ✅ PHASE 6: Add proactive suggestions if appropriate
            if self._should_add_proactive_suggestions(enriched_context):
                enhanced_response = await self._add_proactive_suggestions(enhanced_response)
            
            # ✅ PHASE 7: Add learning acknowledgments
            if insights:
                enhanced_response = await self._add_learning_acknowledgments(enhanced_response, insights)
            
            # Add intelligent follow-up questions
            follow_up_questions = enriched_context.get("follow_up_questions", [])
            if follow_up_questions:
                enhanced_response = self._enhance_response_with_questions(enhanced_response, follow_up_questions)
            
            # Store interaction with enhanced data
            await self.store_enhanced_interaction(message, enhanced_response, enriched_context)
            
            return enhanced_response
            
        except Exception as e:
            error_msg = f"I encountered an error processing your request: {str(e)}"
            log_structured("enhanced_process_message_failed", agent_id=self.agent_id, error=str(e))
            return await self.generate_empathetic_error_response(message, str(e))

    async def enhance_response_with_full_intelligence(self, base_response: str, user_message: str, context: Dict[str, Any]) -> str:
        """Enhanced LLM response generation with full intelligence features"""
        try:
            # Detect emotional tone
            emotional_tone = self._detect_emotional_tone(user_message, context)
            
            # Build comprehensive response context
            response_context = ResponseContext(
                user_message=user_message,
                agent_type=self.agent_type,
                raw_data={
                    "response": base_response,
                    "learned_insights": [insight.content for insight in self.learned_insights[-3:]],
                    "emotional_tone": emotional_tone.value,
                    "conversation_context": self.conversation_context,
                    "proactive_suggestions": len(self.proactive_suggestions)
                },
                user_preferences=context.get('user_preferences', {}),
                communication_style=context.get('communication_style', 'friendly'),
                conversation_history=context.get('conversation_history', ''),
                time_context=context.get('time_context', ''),
                intent=context.get('intent', 'general')
            )
            
            # Apply advanced tone adaptation
            shared_context = context.get('shared_context')
            dynamic_response = await tone_adapter.generate_response(response_context, shared_context)
            
            # Add emotional intelligence
            emotionally_aware_response = await self._add_emotional_intelligence(
                dynamic_response, emotional_tone, context
            )
            
            log_structured("full_intelligence_enhancement_applied",
                         agent_id=self.agent_id,
                         original_length=len(base_response),
                         enhanced_length=len(emotionally_aware_response),
                         emotional_tone=emotional_tone.value,
                         insights_used=len(self.learned_insights),
                         communication_style=context.get('communication_style', 'friendly'))
            
            return emotionally_aware_response
            
        except Exception as e:
            log_structured("full_intelligence_enhancement_failed", agent_id=self.agent_id, error=str(e))
            return base_response

    async def _enrich_context_with_intelligence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich context with learned intelligence and patterns"""
        enriched_context = context.copy()
        
        # Add learned insights
        enriched_context['learned_insights'] = [
            insight.content for insight in self.learned_insights[-5:]
        ]
        
        # Add behavioral patterns
        enriched_context['behavioral_patterns'] = self.pattern_miner.behavioral_patterns
        
        # Add conversation context
        enriched_context['conversation_context'] = self.conversation_context
        
        # Add emotional state tracking
        enriched_context['emotional_state'] = self.emotional_state_tracker
        
        # Add recent vector memory context if available
        if self.vector_memory:
            try:
                recent_memories = await self.vector_memory.query("recent interactions")
                enriched_context['recent_memories'] = recent_memories
            except Exception as e:
                log_structured("memory_query_failed", agent_id=self.agent_id, error=str(e))
        
        return enriched_context

    def _detect_emotional_tone(self, message: str, context: Dict[str, Any]) -> EmotionalTone:
        """Enhanced emotional tone detection"""
        message_lower = message.lower()
        
        # Urgency indicators
        if any(word in message_lower for word in ["urgent", "quickly", "asap", "hurry"]):
            return EmotionalTone.CONCERNED
        
        # Positive indicators
        if any(word in message_lower for word in ["great", "awesome", "love", "excited", "fantastic"]):
            return EmotionalTone.EXCITED
        
        # Happiness indicators
        if any(word in message_lower for word in ["happy", "good", "nice", "wonderful"]):
            return EmotionalTone.HAPPY
        
        # Professional context
        if context.get('communication_style') == 'formal':
            return EmotionalTone.PROFESSIONAL
        
        # Casual context
        if context.get('communication_style') == 'casual':
            return EmotionalTone.CASUAL
        
        # Default helpful tone
        return EmotionalTone.HELPFUL

    def _detect_emotional_indicators(self, message: str) -> Dict[str, float]:
        """Detect emotional indicators in user message"""
        indicators = {
            'urgency': 0.0,
            'positivity': 0.0,
            'formality': 0.0,
            'casualness': 0.0
        }
        
        message_lower = message.lower()
        
        # Urgency
        urgency_words = ["urgent", "quickly", "asap", "hurry", "immediately"]
        indicators['urgency'] = sum(1 for word in urgency_words if word in message_lower) / len(urgency_words)
        
        # Positivity
        positive_words = ["great", "awesome", "love", "excited", "fantastic", "wonderful", "happy"]
        indicators['positivity'] = sum(1 for word in positive_words if word in message_lower) / len(positive_words)
        
        # Formality
        formal_words = ["please", "thank you", "could you", "would you", "kindly"]
        indicators['formality'] = sum(1 for word in formal_words if word in message_lower) / len(formal_words)
        
        # Casualness
        casual_words = ["hey", "hi", "cool", "yeah", "ok", "awesome", "dude"]
        indicators['casualness'] = sum(1 for word in casual_words if word in message_lower) / len(casual_words)
        
        return indicators

    async def _add_emotional_intelligence(self, response: str, emotional_tone: EmotionalTone, context: Dict[str, Any]) -> str:
        """Add emotional intelligence to response"""
        if emotional_tone == EmotionalTone.CONCERNED:
            return f"I understand this seems urgent. {response} Let me prioritize this for you."
        elif emotional_tone == EmotionalTone.EXCITED:
            return f"I can sense your excitement! {response} 🎉"
        elif emotional_tone == EmotionalTone.HAPPY:
            return f"I'm glad you're feeling positive! {response} 😊"
        elif emotional_tone == EmotionalTone.PROFESSIONAL:
            return f"{response}"  # Keep professional tone
        elif emotional_tone == EmotionalTone.CASUAL:
            return f"Hey! {response} Hope that helps! 👍"
        else:
            return response

    async def _add_proactive_suggestions(self, response: str) -> str:
        """Add proactive suggestions to response"""
        active_suggestions = [s for s in self.proactive_suggestions if not s.is_expired()]
        
        if not active_suggestions:
            return response
        
        # Get highest priority suggestion
        top_suggestion = max(active_suggestions, key=lambda s: s.priority)
        
        enhanced_response = f"{response}\n\n💡 **Proactive Suggestion:** {top_suggestion.suggestion}"
        
        # Remove used suggestion
        self.proactive_suggestions = [s for s in self.proactive_suggestions if s != top_suggestion]
        
        return enhanced_response

    async def _add_learning_acknowledgments(self, response: str, insights: List[LearningInsight]) -> str:
        """Add learning acknowledgments to response"""
        if not insights:
            return response
        
        # Select most relevant insight
        most_relevant = max(insights, key=lambda i: i.confidence)
        
        acknowledgment = f"🧠 I've learned that {most_relevant.content.lower()}"
        
        return f"{response}\n\n{acknowledgment}"

    def _should_add_proactive_suggestions(self, context: Dict[str, Any]) -> bool:
        """Determine if proactive suggestions should be added"""
        # Add suggestions if user seems open to them
        return (
            context.get('communication_style') in ['friendly', 'casual'] and
            len(self.proactive_suggestions) > 0 and
            context.get('intent') != 'urgent'
        )

    async def generate_empathetic_error_response(self, original_message: str, error: str) -> str:
        """Generate empathetic error response"""
        return f"""I apologize, but I encountered an issue while processing your request: "{original_message}"

I'm still learning and improving my capabilities. Let me try a different approach to help you.

**What I can still do for you:**
• Provide general assistance and conversation
• Help with basic tasks within my current capabilities
• Learn from this interaction to improve future responses

Would you like me to try helping you in a different way? I'm here to assist you despite this temporary setback."""

    def _enhance_response_with_questions(self, base_response: str, follow_up_questions: List) -> str:
        """Enhanced follow-up question integration with intelligence"""
        if not follow_up_questions:
            return base_response
        
        enhanced_response = base_response
        
        if len(follow_up_questions) == 1:
            enhanced_response += f"\n\n💭 **To help you further:** {follow_up_questions[0].question}"
        else:
            enhanced_response += "\n\n💭 **To better assist you:**"
            for i, question in enumerate(follow_up_questions[:2], 1):
                enhanced_response += f"\n{i}. {question.question}"
        
        return enhanced_response

    def set_session_context(self, session_id: str):
        """Set session context for memory operations"""
        self.current_session_id = session_id
        
        # Reset session-specific data
        self.conversation_context = {'session_id': session_id, 'started_at': time.time()}
        self.emotional_state_tracker = {}
        
        log_structured("enhanced_agent_session_context_set", 
                     agent_id=self.agent_id, 
                     session_id=session_id,
                     intelligence_features_enabled=True)

    async def store_enhanced_interaction(self, user_input: str, response: str, context: Dict[str, Any]):
        """Enhanced interaction storage with intelligence data"""
        try:
            # Enhanced data structure with intelligence information
            interaction_data = {
                "user_input": user_input,
                "agent_response": response,
                "timestamp": time.time(),
                "autogen_version": "0.6.2-official",
                "agent_type": self.agent_type,
                "intelligence_data": {
                    "learned_insights": [insight.content for insight in self.learned_insights[-3:]],
                    "emotional_indicators": self.conversation_context.get('emotional_indicators', {}),
                    "proactive_suggestions_count": len(self.proactive_suggestions),
                    "context_enrichment_applied": True,
                    "pattern_learning_active": True
                },
                "context_used": {
                    "communication_style": context.get('communication_style', 'friendly'),
                    "user_preferences": context.get('user_preferences', {}),
                    "intent": context.get('intent', 'general'),
                    "conversation_continuity": len(context.get('conversation_history', '')) > 0
                }
            }
            
            interaction_id = self.database.store_agent_interaction(
                agent_id=self.agent_id,
                interaction_type="enhanced_chat",
                data=interaction_data
            )

            # Enhanced session storage
            if self.current_session_id:
                chat_id = self.database.store_chat_session(
                    session_id=self.current_session_id,
                    user_message=user_input,
                    agent_response=response,
                    model_used="ollama-enhanced"
                )
                
                if chat_id:
                    log_structured("enhanced_chat_session_stored",
                                agent_id=self.agent_id,
                                session_id=self.current_session_id,
                                chat_id=chat_id,
                                intelligence_features_used=True)
            
            # Enhanced vector memory storage
            if self.vector_memory and self.current_session_id:
                # Store interaction with rich metadata
                interaction_content = f"""User: {user_input}
{self.agent_id}: {response}
Context: {json.dumps(context.get('user_preferences', {}), indent=2)}
Insights: {[insight.content for insight in self.learned_insights[-2:]]}
Emotional Tone: {self.conversation_context.get('emotional_indicators', {})}"""
                
                memory_content = MemoryContent(
                    content=interaction_content,
                    mime_type=MemoryMimeType.TEXT
                )
                await self.vector_memory.add(memory_content)
                
        except Exception as e:
            log_structured("enhanced_interaction_storage_failed", agent_id=self.agent_id, error=str(e))

    async def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive intelligence summary"""
        return {
            "agent_id": self.agent_id,
            "intelligence_features": {
                "pattern_learning": True,
                "emotional_intelligence": True,
                "proactive_suggestions": True,
                "context_awareness": True,
                "dynamic_responses": True
            },
            "learned_insights": len(self.learned_insights),
            "active_suggestions": len([s for s in self.proactive_suggestions if not s.is_expired()]),
            "conversation_context": self.conversation_context,
            "behavioral_patterns": self.pattern_miner.behavioral_patterns,
            "session_id": self.current_session_id,
            "memory_available": self.vector_memory is not None
        }


class EnhancedAgentRegistry:
    """Enhanced Agent Registry with full intelligence tracking"""
    
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.intelligence_metrics: Dict[str, Dict[str, Any]] = {}
        self.system_intelligence_summary: Dict[str, Any] = {}
    
    def register_agent(self, agent: AgentBase) -> bool:
        """Register an agent with enhanced intelligence tracking"""
        try:
            agent_key = getattr(agent, 'agent_id', None) or getattr(agent, 'name', 'unknown')
            agent_type = getattr(agent, 'agent_type', 'unknown')
            
            # Store agent
            self.agents[agent_key] = agent
            
            # Track intelligence capabilities
            self.intelligence_metrics[agent_key] = {
                'type': agent_type,
                'dynamic_responses': hasattr(agent, 'enhance_response_with_full_intelligence'),
                'pattern_learning': hasattr(agent, 'pattern_miner'),
                'emotional_intelligence': hasattr(agent, 'emotional_state_tracker'),
                'proactive_suggestions': hasattr(agent, 'proactive_suggestions'),
                'context_awareness': hasattr(agent, 'conversation_context'),
                'memory_enabled': hasattr(agent, 'vector_memory') and agent.vector_memory is not None,
                'registered_at': time.time(),
                'intelligence_level': 'enhanced'
            }
            
            # Update system summary
            self._update_system_intelligence_summary()
            
            log_structured("enhanced_autogen_062_agent_registered",
                         agent_name=agent_key,
                         agent_type=agent_type,
                         intelligence_capabilities=self.intelligence_metrics[agent_key],
                         autogen_version="0.6.2-official")
            return True
            
        except Exception as e:
            log_structured("enhanced_agent_registration_failed", 
                         agent_name=getattr(agent, 'name', 'unknown'),
                         error=str(e))
            return False
    
    def _update_system_intelligence_summary(self):
        """Update system-wide intelligence summary"""
        total_agents = len(self.agents)
        if total_agents == 0:
            return
        
        intelligence_counts = {
            'dynamic_responses': sum(1 for m in self.intelligence_metrics.values() if m.get('dynamic_responses')),
            'pattern_learning': sum(1 for m in self.intelligence_metrics.values() if m.get('pattern_learning')),
            'emotional_intelligence': sum(1 for m in self.intelligence_metrics.values() if m.get('emotional_intelligence')),
            'proactive_suggestions': sum(1 for m in self.intelligence_metrics.values() if m.get('proactive_suggestions')),
            'context_awareness': sum(1 for m in self.intelligence_metrics.values() if m.get('context_awareness')),
            'memory_enabled': sum(1 for m in self.intelligence_metrics.values() if m.get('memory_enabled'))
        }
        
        self.system_intelligence_summary = {
            'total_agents': total_agents,
            'intelligence_coverage': {
                feature: f"{count}/{total_agents} ({count/total_agents*100:.1f}%)"
                for feature, count in intelligence_counts.items()
            },
            'system_intelligence_level': 'enhanced' if all(
                count == total_agents for count in intelligence_counts.values()
            ) else 'mixed',
            'last_updated': time.time()
        }
    
    def get_agent(self, agent_id: str) -> Optional[AgentBase]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all registered agent IDs"""
        return list(self.agents.keys())
    
    def get_intelligence_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get intelligence metrics for specific agent"""
        return self.intelligence_metrics.get(agent_id, {})
    
    def get_system_intelligence_summary(self) -> Dict[str, Any]:
        """Get system-wide intelligence summary"""
        return self.system_intelligence_summary
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including intelligence metrics"""
        agent_summaries = {}
        
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_intelligence_summary'):
                agent_summaries[agent_id] = await agent.get_intelligence_summary()
        
        return {
            'system_overview': self.system_intelligence_summary,
            'agent_intelligence_metrics': self.intelligence_metrics,
            'agent_summaries': agent_summaries,
            'autogen_version': '0.6.2-official',
            'enhancement_level': 'full_intelligence',
            'timestamp': time.time()
        }
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from registry"""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                if agent_id in self.intelligence_metrics:
                    del self.intelligence_metrics[agent_id]
                self._update_system_intelligence_summary()
                log_structured("enhanced_agent_removed", agent_id=agent_id)
                return True
            return False
        except Exception as e:
            log_structured("enhanced_agent_removal_failed", agent_id=agent_id, error=str(e))
            return False


# Global enhanced agent registry
agentregistry = EnhancedAgentRegistry()
