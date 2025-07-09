# src/intent_classifier.py
"""
Intent Classification System for Jarvis-AI Intelligent Edition
Compatible with AutoGen 0.6.2 architecture
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.agents.core.logging_config import log_structured
from config.settings import settings

@dataclass
class IntentTemplate:
    """Intent template with examples and confidence threshold"""
    name: str
    examples: List[str]
    confidence_threshold: float = 0.7
    target_agent: str = "orchestrator"
    requires_context: bool = False

class IntentClassifier:
    """
    Local intent classification using sentence transformers
    Replaces keyword-based routing with semantic understanding
    """
    
    def __init__(self):
        self.model = None
        self.intent_templates: Dict[str, IntentTemplate] = {}
        self.intent_embeddings: Dict[str, np.ndarray] = {}
        self.initialized = False
        self.intent_cache = {}
        self.last_message = None
        self.last_result = None
        
    async def initialize(self) -> bool:
        """Initialize the intent classifier with local model"""
        try:
            # Use lightweight sentence transformer for local processing
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Define intent templates with examples
            self.intent_templates = {
                "weather_query": IntentTemplate(
                    name="weather_query",
                    examples=[
                        "What's the weather like?",
                        "Current temperature in Brussels",
                        "Is it going to rain today?",
                        "Weather forecast for tomorrow",
                        "How hot is it outside?",
                        "Check the weather conditions"
                    ],
                    confidence_threshold=0.6,
                    target_agent="weather"
                ),
                "routine_create": IntentTemplate(
                    name="routine_create",
                    examples=[
                        "Create a morning routine",
                        "Help me build a workout schedule",
                        "Make a new evening routine",
                        "I want to start a daily habit",
                        "Design a productivity routine",
                        "Set up my morning activities"
                    ],
                    confidence_threshold=0.65,
                    target_agent="routine",
                    requires_context=True
                ),
                "routine_manage": IntentTemplate(
                    name="routine_manage",
                    examples=[
                        "Show my routines",
                        "List my current habits",
                        "Track my progress",
                        "I completed my routine",
                        "Update my schedule",
                        "Modify my morning routine"
                    ],
                    confidence_threshold=0.6,
                    target_agent="routine"
                ),
                "system_query": IntentTemplate(
                    name="system_query",
                    examples=[
                        "System status",
                        "Check agent health",
                        "How are you doing?",
                        "Show system information",
                        "Agent diagnostics",
                        "Performance metrics"
                    ],
                    confidence_threshold=0.7,
                    target_agent="orchestrator"
                ),
                "general_conversation": IntentTemplate(
                    name="general_conversation",
                    examples=[
                        "Hello, how are you?",
                        "What can you help me with?",
                        "Tell me about your capabilities",
                        "I need assistance",
                        "Help me understand",
                        "What should I do today?"
                    ],
                    confidence_threshold=0.5,
                    target_agent="orchestrator"
                )
            }
            
            # Pre-compute embeddings for all intent examples
            await self._compute_intent_embeddings()
            
            self.initialized = True
            log_structured("intent_classifier_initialized", 
                         model="all-MiniLM-L6-v2", 
                         intents=list(self.intent_templates.keys()),
                         autogen_version="0.6.2-official")
            return True
            
        except Exception as e:
            log_structured("intent_classifier_init_failed", error=str(e))
            return False
    
    async def _compute_intent_embeddings(self):
        """Pre-compute embeddings for all intent examples"""
        for intent_name, template in self.intent_templates.items():
            # Combine all examples into a single representation
            combined_examples = " ".join(template.examples)
            embedding = self.model.encode(combined_examples)
            self.intent_embeddings[intent_name] = embedding
    
    async def classify_intent(self, message: str) -> Tuple[str, float, str]:
        if not self.initialized:
            await self.initialize()
        
        try:
            message_lower = message.lower().strip()
            
            # Add keyword fallbacks for obvious cases (CRITICAL)
            keyword_mapping = {
                "weather": ("weather_query", 0.9, "weather"),
                "temperature": ("weather_query", 0.9, "weather"),
                "forecast": ("weather_query", 0.9, "weather"),
                "rain": ("weather_query", 0.85, "weather"),
                "sunny": ("weather_query", 0.85, "weather"),
                "routine": ("routine_manage", 0.9, "routine"),
                "create routine": ("routine_create", 0.95, "routine"),
                "morning routine": ("routine_create", 0.95, "routine"),
                "status": ("system_query", 0.9, "orchestrator"),
                "system": ("system_query", 0.9, "orchestrator")
            }
            
            # Check for keyword matches first
            for keyword, (intent, confidence, agent) in keyword_mapping.items():
                if keyword in message_lower:
                    log_structured("intent_classified_by_keyword", 
                                message=message[:100],
                                keyword=keyword,
                                intent=intent,
                                confidence=confidence)
                    return intent, confidence, agent
            
            # Fallback to semantic similarity
            message_embedding = self.model.encode(message_lower)
            
            best_intent = "general_conversation"
            best_confidence = 0.0
            best_agent = "orchestrator"
            
            for intent_name, template in self.intent_templates.items():
                max_similarity = 0.0
                for example in template.examples:
                    example_embedding = self.model.encode(example.lower())
                    similarity = cosine_similarity(
                        [message_embedding], 
                        [example_embedding]
                    )[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                if max_similarity > template.confidence_threshold and max_similarity > best_confidence:
                    best_intent = intent_name
                    best_confidence = max_similarity
                    best_agent = template.target_agent
            
            log_structured("intent_classified", 
                        message=message[:100],
                        intent=best_intent,
                        confidence=best_confidence,
                        target_agent=best_agent)
            
            return best_intent, best_confidence, best_agent
            
        except Exception as e:
            log_structured("intent_classification_failed", error=str(e))
            return "general_conversation", 0.5, "orchestrator"
    

        
    def get_intent_info(self, intent_name: str) -> Optional[IntentTemplate]:
        """Get information about a specific intent"""
        return self.intent_templates.get(intent_name)

# Global intent classifier instance
intent_classifier = IntentClassifier()
