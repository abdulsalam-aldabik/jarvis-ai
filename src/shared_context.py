"""
Shared Message Context for AutoGen 0.6.2 Multi-Agent Coordination
Enables cross-agent memory sharing and conversation continuity
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
import json
from src.agents.core.logging_config import log_structured

@dataclass
class ConversationTurn:
    """Individual conversation turn with full context"""
    user_message: str
    agent_response: str
    agent_id: str
    intent: str
    confidence: float
    timestamp: float
    context_used: Dict[str, Any]
    user_preferences_learned: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SharedMessageContext:
    """Global context shared across all agents in a session"""
    session_id: str
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    pending_follow_ups: List[str] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    current_topic: Optional[str] = None
    conversation_theme: Optional[str] = None
    user_communication_style: str = "friendly"
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn to shared context"""
        self.conversation_turns.append(turn)
        
        # Auto-detect conversation themes
        self._update_conversation_theme(turn)
        
        # Learn from user communication patterns
        self._analyze_communication_style(turn.user_message)
        
        log_structured("shared_context_turn_added",
                     session_id=self.session_id[:8],
                     agent_id=turn.agent_id,
                     intent=turn.intent,
                     total_turns=len(self.conversation_turns))
    
    def update_preferences(self, new_preferences: Dict[str, Any], source_agent: str = "system"):
        """Update user preferences with learning acknowledgment"""
        changes_made = []
        
        for key, value in new_preferences.items():
            if key not in self.user_preferences or self.user_preferences[key] != value:
                old_value = self.user_preferences.get(key, "not set")
                self.user_preferences[key] = value
                changes_made.append(f"{key}: {old_value} → {value}")
                
                # Generate learning acknowledgment
                acknowledgment = self._generate_learning_acknowledgment(key, value)
                self.pending_follow_ups.append(acknowledgment)
        
        if changes_made:
            log_structured("preferences_updated",
                         session_id=self.session_id[:8],
                         source_agent=source_agent,
                         changes=changes_made,
                         total_preferences=len(self.user_preferences))
    
    def get_conversation_summary(self, last_n_turns: int = 3) -> str:
        """Get summary of recent conversation for context"""
        recent_turns = self.conversation_turns[-last_n_turns:]
        return "\n".join([
            f"User: {turn.user_message}\n{turn.agent_id}: {turn.agent_response}" 
            for turn in recent_turns
        ])
    
    def get_context_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get relevant context for a specific agent"""
        # Get turns involving this agent
        agent_turns = [t for t in self.conversation_turns if t.agent_id == agent_id]
        
        # Get cross-agent context
        other_agent_context = [
            f"{t.agent_id} discussed: {t.user_message}" 
            for t in self.conversation_turns[-3:] 
            if t.agent_id != agent_id
        ]
        
        return {
            "user_preferences": self.user_preferences,
            "conversation_theme": self.conversation_theme,
            "communication_style": self.user_communication_style,
            "recent_context": self.get_conversation_summary(3),
            "agent_specific_turns": len(agent_turns),
            "cross_agent_context": other_agent_context,
            "pending_follow_ups": self.pending_follow_ups
        }
    
    def _update_conversation_theme(self, turn: ConversationTurn):
        """Auto-detect and update conversation themes"""
        themes = {
            "weather": ["weather", "temperature", "forecast", "rain", "sunny"],
            "routines": ["routine", "habit", "schedule", "daily", "morning"],
            "system": ["status", "health", "check", "system"],
            "general": ["hello", "hi", "help", "thanks"]
        }
        
        message_lower = turn.user_message.lower()
        for theme, keywords in themes.items():
            if any(keyword in message_lower for keyword in keywords):
                if self.conversation_theme != theme:
                    self.conversation_theme = theme
                    log_structured("conversation_theme_updated",
                                 session_id=self.session_id[:8],
                                 new_theme=theme,
                                 trigger_message=turn.user_message[:50])
                break
    
    def _analyze_communication_style(self, user_message: str):
        """Analyze and adapt to user communication style"""
        message_lower = user_message.lower()
        
        # Detect communication patterns
        if any(word in message_lower for word in ["please", "thank you", "could you"]):
            style = "formal"
        elif any(word in message_lower for word in ["yo", "hey", "sup", "cool"]):
            style = "casual"
        elif "!" in user_message or any(word in message_lower for word in ["awesome", "great", "amazing"]):
            style = "enthusiastic"
        else:
            style = "friendly"
        
        if self.user_communication_style != style:
            self.user_communication_style = style
            log_structured("communication_style_detected",
                         session_id=self.session_id[:8],
                         style=style,
                         message=user_message[:50])
    
    def _generate_learning_acknowledgment(self, key: str, value: Any) -> str:
        """Generate natural learning acknowledgments"""
        acknowledgments = {
            "preferred_time": f"Got it! I'll remember you prefer {value} timing for activities.",
            "communication_style": f"I'll adapt my responses to be more {value}.",
            "routine_type": f"Noted - you're interested in {value} routines.",
            "activity_preferences": f"I'll keep in mind that you enjoy {value}.",
            "default": f"I've learned that you prefer {key}: {value}"
        }
        
        return acknowledgments.get(key, acknowledgments["default"])

class SharedContextManager:
    """Manages shared contexts across sessions"""
    
    def __init__(self):
        self.active_contexts: Dict[str, SharedMessageContext] = {}
    
    def get_or_create_context(self, session_id: str) -> SharedMessageContext:
        """Get existing context or create new one for session"""
        if session_id not in self.active_contexts:
            self.active_contexts[session_id] = SharedMessageContext(session_id=session_id)
            log_structured("shared_context_created",
                         session_id=session_id[:8],
                         total_active_sessions=len(self.active_contexts))
        
        return self.active_contexts[session_id]
    
    def cleanup_old_contexts(self, max_age_hours: int = 24):
        """Clean up old contexts to prevent memory bloat"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_sessions = []
        for session_id, context in self.active_contexts.items():
            if context.conversation_turns:
                last_turn_time = context.conversation_turns[-1].timestamp
                if current_time - last_turn_time > max_age_seconds:
                    old_sessions.append(session_id)
        
        for session_id in old_sessions:
            del self.active_contexts[session_id]
            log_structured("shared_context_cleaned",
                         session_id=session_id[:8],
                         remaining_sessions=len(self.active_contexts))

# Global shared context manager
shared_context_manager = SharedContextManager()
