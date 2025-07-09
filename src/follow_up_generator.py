"""
Follow-up Question Generator for Context-Aware Conversation
Generates intelligent clarifying questions based on intent and context
"""

from typing import List, Dict, Any, Optional
from src.agents.core.logging_config import log_structured

class FollowUpGenerator:
    """Generates contextual follow-up questions to improve conversation flow"""
    
    def __init__(self):
        self.question_templates = {
            "routine_create": [
                "What time of day would work best for this routine?",
                "How long would you like to spend on this activity?",
                "Are there any specific goals you want to achieve?",
                "Should this be a daily routine or specific days?"
            ],
            "weather_query": [
                "Are you planning any outdoor activities?",
                "Would you like a forecast for specific days?",
                "Are you concerned about any particular weather conditions?"
            ],
            "general_conversation": [
                "What specific area would you like help with?",
                "Is there anything particular you'd like to know more about?",
                "How can I best assist you today?"
            ]
        }
    
    def generate_follow_ups(self, intent: str, user_message: str, 
                           conversation_history: List[Dict], 
                           shared_context: Any) -> List[str]:
        """Generate relevant follow-up questions based on context"""
        
        # Check if we need more information for this intent
        missing_info = self._analyze_missing_information(intent, user_message, shared_context)
        
        if not missing_info:
            return []
        
        # Generate context-specific questions
        questions = self._generate_context_questions(intent, missing_info, shared_context)
        
        # Limit to 1-2 most relevant questions
        priority_questions = self._prioritize_questions(questions, shared_context)
        
        if priority_questions:
            log_structured("follow_up_questions_generated",
                         intent=intent,
                         question_count=len(priority_questions),
                         missing_info=missing_info)
        
        return priority_questions
    
    def _analyze_missing_information(self, intent: str, message: str, shared_context: Any) -> List[str]:
        """Analyze what information is missing for better assistance"""
        missing = []
        
        if intent == "routine_create":
            if "time" not in message.lower():
                missing.append("time_preference")
            if not any(word in message.lower() for word in ["daily", "weekly", "morning", "evening"]):
                missing.append("frequency")
            if len(message.split()) < 5:  # Very short request
                missing.append("details")
        
        elif intent == "weather_query":
            if not any(word in message.lower() for word in ["today", "tomorrow", "week", "weekend"]):
                missing.append("timeframe")
            if "location" not in shared_context.user_preferences:
                missing.append("location")
        
        return missing
    
    def _generate_context_questions(self, intent: str, missing_info: List[str], 
                                   shared_context: Any) -> List[str]:
        """Generate questions based on missing information"""
        questions = []
        
        question_map = {
            "routine_create": {
                "time_preference": "What time of day works best for you?",
                "frequency": "How often would you like to do this - daily or specific days?",
                "details": "Could you tell me more about what you'd like to include in this routine?"
            },
            "weather_query": {
                "timeframe": "Are you asking about today, tomorrow, or a specific time period?",
                "location": "What location would you like the weather for?"
            }
        }
        
        if intent in question_map:
            for info in missing_info:
                if info in question_map[intent]:
                    questions.append(question_map[intent][info])
        
        return questions
    
    def _prioritize_questions(self, questions: List[str], shared_context: Any) -> List[str]:
        """Prioritize questions based on conversation context"""
        # Limit to 1-2 most important questions
        return questions[:2]

# Global follow-up generator
follow_up_generator = FollowUpGenerator()
