"""
Follow-up Question Generator for Jarvis-AI Intelligent Edition
Generates contextual follow-up questions to improve conversation flow
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.agents.core.logging_config import log_structured
from src.shared_context import SharedMessageContext

class QuestionType(Enum):
    """Types of follow-up questions"""
    CLARIFICATION = "clarification"
    SPECIFICATION = "specification"
    PREFERENCE = "preference"
    CONFIRMATION = "confirmation"
    EXPANSION = "expansion"

@dataclass
class FollowUpQuestion:
    """Structure for generated follow-up questions"""
    question: str
    question_type: QuestionType
    priority: int  # 1-5, higher = more important
    context_needed: str
    examples: List[str] = None

class FollowUpGenerator:
    """
    Generates intelligent follow-up questions based on conversation context
    """
    
    def __init__(self):
        self.question_templates = self._initialize_question_templates()
        self.intent_analyzers = self._initialize_intent_analyzers()
    
    async def analyze_and_generate(self, 
                                 user_message: str,
                                 intent: str,
                                 confidence: float,
                                 shared_context: SharedMessageContext) -> List[FollowUpQuestion]:
        """
        Analyze conversation context and generate relevant follow-up questions
        """
        try:
            # Get context analysis
            context_gaps = self._identify_context_gaps(intent, user_message, shared_context)
            
            # Generate questions based on gaps
            questions = []
            for gap in context_gaps:
                question = self._generate_question_for_gap(gap, intent, shared_context)
                if question:
                    questions.append(question)
            
            # Prioritize and limit questions
            prioritized_questions = self._prioritize_questions(questions, shared_context)
            
            log_structured("follow_up_questions_generated",
                         session_id=shared_context.session_id[:8],
                         intent=intent,
                         questions_count=len(prioritized_questions),
                         context_gaps=len(context_gaps))
            
            return prioritized_questions[:2]  # Limit to 2 questions max
            
        except Exception as e:
            log_structured("follow_up_generation_failed", error=str(e))
            return []
    
    def _identify_context_gaps(self, 
                             intent: str, 
                             user_message: str, 
                             shared_context: SharedMessageContext) -> List[str]:
        """Identify missing context information"""
        gaps = []
        
        if intent == "routine_create":
            gaps.extend(self._analyze_routine_gaps(user_message, shared_context))
        elif intent == "weather_query":
            gaps.extend(self._analyze_weather_gaps(user_message, shared_context))
        elif intent == "general_conversation":
            gaps.extend(self._analyze_general_gaps(user_message, shared_context))
        elif intent == "system_query":
            gaps.extend(self._analyze_system_gaps(user_message, shared_context))
        
        return gaps
    
    def _analyze_routine_gaps(self, user_message: str, shared_context: SharedMessageContext) -> List[str]:
        """Analyze missing information for routine creation"""
        gaps = []
        message_lower = user_message.lower()
        
        # Check for missing time specification
        time_keywords = ["morning", "afternoon", "evening", "night", "am", "pm"]
        if not any(keyword in message_lower for keyword in time_keywords):
            gaps.append("time_preference")
        
        # Check for missing activity type
        activity_keywords = ["workout", "exercise", "meditation", "reading", "work"]
        if not any(keyword in message_lower for keyword in activity_keywords):
            gaps.append("activity_type")
        
        # Check for missing duration
        duration_keywords = ["minute", "hour", "quick", "long", "short"]
        if not any(keyword in message_lower for keyword in duration_keywords):
            gaps.append("duration")
        
        # Check for missing goal
        goal_keywords = ["fitness", "productivity", "relaxation", "health", "energy"]
        if not any(keyword in message_lower for keyword in goal_keywords):
            gaps.append("goal")
        
        # Check user preferences for known patterns
        if "preferred_routine_times" not in shared_context.user_preferences:
            gaps.append("time_preference_learning")
        
        return gaps
    
    def _analyze_weather_gaps(self, user_message: str, shared_context: SharedMessageContext) -> List[str]:
        """Analyze missing information for weather queries"""
        gaps = []
        message_lower = user_message.lower()
        
        # Check for missing time frame
        time_keywords = ["today", "tomorrow", "week", "weekend", "tonight"]
        if not any(keyword in message_lower for keyword in time_keywords):
            gaps.append("time_frame")
        
        # Check for missing location (if not in preferences)
        if "location" not in shared_context.user_preferences:
            location_keywords = ["in", "at", "for"]
            if not any(keyword in message_lower for keyword in location_keywords):
                gaps.append("location")
        
        # Check for missing weather aspect
        aspect_keywords = ["temperature", "rain", "snow", "wind", "humidity"]
        if len(message_lower.split()) <= 2 and not any(keyword in message_lower for keyword in aspect_keywords):
            gaps.append("weather_aspect")
        
        return gaps
    
    def _analyze_general_gaps(self, user_message: str, shared_context: SharedMessageContext) -> List[str]:
        """Analyze gaps in general conversation"""
        gaps = []
        message_lower = user_message.lower()
        
        # Check for vague requests
        if len(message_lower.split()) <= 2:
            gaps.append("clarification")
        
        # Check for help requests without specifics
        if "help" in message_lower and len(message_lower.split()) <= 3:
            gaps.append("help_specification")
        
        # First-time user detection
        if len(shared_context.conversation_turns) <= 1:
            gaps.append("user_introduction")
        
        return gaps
    
    def _analyze_system_gaps(self, user_message: str, shared_context: SharedMessageContext) -> List[str]:
        """Analyze gaps in system queries"""
        gaps = []
        message_lower = user_message.lower()
        
        # Check for specific system aspect
        if message_lower in ["status", "check", "system"]:
            gaps.append("system_aspect")
        
        return gaps
    
    def _generate_question_for_gap(self, 
                                 gap: str, 
                                 intent: str, 
                                 shared_context: SharedMessageContext) -> Optional[FollowUpQuestion]:
        """Generate a specific question for identified gap"""
        
        # Get question template for this gap
        template = self.question_templates.get(gap)
        if not template:
            return None
        
        # Personalize question based on context
        question_text = self._personalize_question(template, shared_context)
        
        return FollowUpQuestion(
            question=question_text,
            question_type=QuestionType.CLARIFICATION,
            priority=template.get("priority", 3),
            context_needed=gap,
            examples=template.get("examples", [])
        )
    
    def _personalize_question(self, template: Dict[str, Any], shared_context: SharedMessageContext) -> str:
        """Personalize question based on user context and communication style"""
        base_question = template["question"]
        
        # Adapt to communication style
        style = shared_context.user_communication_style
        
        if style == "formal":
            # More formal phrasing
            base_question = base_question.replace("What's", "What is")
            base_question = base_question.replace("you'd", "you would")
        elif style == "casual":
            # More casual phrasing
            base_question = base_question.replace("What is", "What's")
            base_question = base_question.replace("you would", "you'd")
        elif style == "enthusiastic":
            # Add enthusiasm
            if not base_question.endswith("!"):
                base_question = base_question.replace("?", "!")
        
        return base_question
    
    def _prioritize_questions(self, 
                            questions: List[FollowUpQuestion], 
                            shared_context: SharedMessageContext) -> List[FollowUpQuestion]:
        """Prioritize questions based on context and conversation flow"""
        
        # Sort by priority (higher first)
        sorted_questions = sorted(questions, key=lambda q: q.priority, reverse=True)
        
        # Filter out questions that might be redundant
        filtered_questions = []
        recent_questions = [turn.user_message for turn in shared_context.conversation_turns[-3:]]
        
        for question in sorted_questions:
            # Skip if similar question was asked recently
            if not self._is_redundant_question(question.question, recent_questions):
                filtered_questions.append(question)
        
        return filtered_questions
    
    def _is_redundant_question(self, question: str, recent_questions: List[str]) -> bool:
        """Check if question is redundant with recent conversation"""
        question_words = set(question.lower().split())
        
        for recent in recent_questions:
            recent_words = set(recent.lower().split())
            # If significant overlap, consider redundant
            if len(question_words.intersection(recent_words)) > len(question_words) * 0.6:
                return True
        
        return False
    
    def _initialize_question_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize question templates for different context gaps"""
        return {
            "time_preference": {
                "question": "What time of day works best for you - morning, afternoon, or evening?",
                "priority": 5,
                "examples": ["morning", "afternoon", "evening", "7 AM", "after work"]
            },
            "activity_type": {
                "question": "What kind of activities would you like to include in your routine?",
                "priority": 4,
                "examples": ["exercise", "meditation", "reading", "stretching", "planning"]
            },
            "duration": {
                "question": "How much time would you like to dedicate to this routine?",
                "priority": 3,
                "examples": ["15 minutes", "30 minutes", "1 hour", "quick", "detailed"]
            },
            "goal": {
                "question": "What's your main goal - fitness, productivity, relaxation, or something else?",
                "priority": 4,
                "examples": ["fitness", "productivity", "relaxation", "energy", "focus"]
            },
            "time_frame": {
                "question": "Are you asking about today, tomorrow, or a specific time period?",
                "priority": 3,
                "examples": ["today", "tomorrow", "this weekend", "next week"]
            },
            "location": {
                "question": "Which location would you like weather information for?",
                "priority": 5,
                "examples": ["Brussels", "current location", "home", "work"]
            },
            "weather_aspect": {
                "question": "Are you interested in temperature, precipitation, or general conditions?",
                "priority": 2,
                "examples": ["temperature", "rain", "general forecast", "wind"]
            },
            "clarification": {
                "question": "Could you tell me more about what you'd like help with?",
                "priority": 4,
                "examples": ["routines", "weather", "system status", "general assistance"]
            },
            "help_specification": {
                "question": "I'd be happy to help! Are you interested in weather, routines, system status, or something else?",
                "priority": 4,
                "examples": ["weather information", "routine planning", "system monitoring"]
            },
            "user_introduction": {
                "question": "Nice to meet you! What brings you here today - would you like help with routines, weather, or just exploring what I can do?",
                "priority": 3,
                "examples": ["create routines", "check weather", "learn about features"]
            },
            "system_aspect": {
                "question": "Would you like to see overall system health, agent status, or database information?",
                "priority": 2,
                "examples": ["system health", "agent status", "database stats"]
            },
            "time_preference_learning": {
                "question": "I'll remember your preference! Do you usually prefer morning, afternoon, or evening activities?",
                "priority": 3,
                "examples": ["morning person", "afternoon", "evening", "flexible"]
            }
        }
    
    def _initialize_intent_analyzers(self) -> Dict[str, Any]:
        """Initialize intent-specific analyzers"""
        return {
            "routine_create": self._analyze_routine_gaps,
            "routine_manage": self._analyze_routine_gaps,
            "weather_query": self._analyze_weather_gaps,
            "general_conversation": self._analyze_general_gaps,
            "system_query": self._analyze_system_gaps
        }

# Global follow-up generator instance
follow_up_generator = FollowUpGenerator()
