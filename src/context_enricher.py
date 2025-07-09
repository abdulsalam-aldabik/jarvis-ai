# src/context_enricher.py
"""
Context Enrichment System for Jarvis-AI Intelligent Edition
Merges user profile, history, and environmental context
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.agents.core.database import db_manager
from src.agents.core.logging_config import log_structured
from config.settings import settings

@dataclass
class EnrichedContext:
    """Enriched context object containing all relevant information"""
    user_message: str
    intent: str
    confidence: float
    target_agent: str
    session_id: str
    timestamp: float
    
    # User profile information
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    
    # Environmental context
    time_context: Dict[str, Any]
    system_context: Dict[str, Any]
    
    # Agent-specific context
    agent_memory: List[Dict[str, Any]]
    previous_interactions: List[Dict[str, Any]]

class ContextEnricher:
    """
    Context enrichment engine that merges multiple data sources
    to provide comprehensive situational awareness
    """
    
    def __init__(self):
        self.database = db_manager
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes cache
        
    async def enrich_context(self, 
                           user_message: str,
                           intent: str,
                           confidence: float,
                           target_agent: str,
                           session_id: str) -> EnrichedContext:
        """
        Enrich context with user profile, history, and environmental data
        """
        try:
            timestamp = time.time()
            
            # Gather all context components in parallel
            user_preferences = await self._get_user_preferences(session_id)
            conversation_history = await self._get_conversation_history(session_id)
            time_context = self._get_time_context()
            system_context = await self._get_system_context()
            agent_memory = await self._get_agent_memory(target_agent, session_id)
            previous_interactions = await self._get_previous_interactions(target_agent, session_id)
            
            enriched_context = EnrichedContext(
                user_message=user_message,
                intent=intent,
                confidence=confidence,
                target_agent=target_agent,
                session_id=session_id,
                timestamp=timestamp,
                user_preferences=user_preferences,
                conversation_history=conversation_history,
                time_context=time_context,
                system_context=system_context,
                agent_memory=agent_memory,
                previous_interactions=previous_interactions
            )
            
            log_structured("context_enriched",
                         intent=intent,
                         target_agent=target_agent,
                         session_id=session_id,
                         context_size=len(conversation_history),
                         preferences_count=len(user_preferences))
            
            return enriched_context
            
        except Exception as e:
            log_structured("context_enrichment_failed", error=str(e))
            # Return minimal context on failure
            return EnrichedContext(
                user_message=user_message,
                intent=intent,
                confidence=confidence,
                target_agent=target_agent,
                session_id=session_id,
                timestamp=time.time(),
                user_preferences={},
                conversation_history=[],
                time_context=self._get_time_context(),
                system_context={},
                agent_memory=[],
                previous_interactions=[]
            )
    
    async def _get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Extract user preferences from interaction history"""
        try:
            # Query recent interactions to infer preferences
            recent_interactions = await self._get_recent_interactions(session_id, limit=50)
            
            preferences = {
                "preferred_routine_types": [],
                "preferred_times": {},
                "communication_style": "friendly",
                "activity_preferences": [],
                "location_preferences": {}
            }
            
            # Analyze interactions for preference patterns
            for interaction in recent_interactions:
                data = interaction.get('data', {})
                
                # Extract routine preferences
                if interaction.get('interaction_type') == 'routine_created':
                    routine_type = data.get('type', 'general')
                    if routine_type not in preferences["preferred_routine_types"]:
                        preferences["preferred_routine_types"].append(routine_type)
                
                # Extract time preferences
                if 'time' in data:
                    time_pref = data['time']
                    activity = data.get('activity', 'general')
                    preferences["preferred_times"][activity] = time_pref
                
                # Extract activity preferences
                if 'activities' in data:
                    activities = data['activities']
                    if isinstance(activities, list):
                        preferences["activity_preferences"].extend(activities)
            
            return preferences
            
        except Exception as e:
            log_structured("user_preferences_extraction_failed", error=str(e))
            return {}
    
    async def _get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history using enhanced database method"""
        try:
            # Use the new database method
            history_records = self.database.get_session_history(session_id, limit)
            
            if not history_records:
                # Try alternative: get recent history from any session for this user
                # This helps when session management has issues
                recent_history = self.database.get_recent_chat_history(limit=5)
                log_structured("using_fallback_history", 
                            session_id=session_id[:8],
                            fallback_count=len(recent_history))
                return recent_history
            
            conversation_history = [
                {
                    "user_message": record['user_message'],
                    "agent_response": record['agent_response'],
                    "model_used": record['model_used'],
                    "timestamp": record['created_at'].timestamp()
                }
                for record in history_records
            ]
            
            log_structured("conversation_history_success",
                        session_id=session_id[:8],
                        history_count=len(conversation_history))
            
            return conversation_history
            
        except Exception as e:
            log_structured("conversation_history_retrieval_failed", 
                        session_id=session_id[:8],
                        error=str(e))
            return []

        
    def _get_time_context(self) -> Dict[str, Any]:
        """Get current time context for situational awareness"""
        now = datetime.now()
        
        return {
            "current_time": now.isoformat(),
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "is_morning": 5 <= now.hour <= 11,
            "is_afternoon": 12 <= now.hour <= 17,
            "is_evening": 18 <= now.hour <= 22,
            "is_night": now.hour >= 23 or now.hour <= 4,
            "is_weekend": now.weekday() >= 5,
            "time_period": self._get_time_period(now.hour)
        }
    
    def _get_time_period(self, hour: int) -> str:
        """Get descriptive time period"""
        if 5 <= hour <= 11:
            return "morning"
        elif 12 <= hour <= 17:
            return "afternoon"
        elif 18 <= hour <= 22:
            return "evening"
        else:
            return "night"
    
    async def _get_system_context(self) -> Dict[str, Any]:
        """Get system health and status context"""
        try:
            # Use cached system context if available and fresh
            cache_key = "system_context"
            if cache_key in self.context_cache:
                cached_data = self.context_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['data']
            
            # Get fresh system context
            health_check = self.database.health_check()
            active_agents = self.database.get_active_agents()
            
            system_context = {
                "database_status": health_check.get('status', 'unknown'),
                "active_agents_count": len(active_agents),
                "active_agents": [agent['agent_id'] for agent in active_agents],
                "system_load": "normal",  # Could be enhanced with actual metrics
                "memory_usage": "normal",
                "uptime": "operational"
            }
            
            # Cache the result
            self.context_cache[cache_key] = {
                'data': system_context,
                'timestamp': time.time()
            }
            
            return system_context
            
        except Exception as e:
            log_structured("system_context_retrieval_failed", error=str(e))
            return {"status": "unknown"}
    
    async def _get_agent_memory(self, agent_id: str, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get agent-specific memory context"""
        try:
            agent_context = self.database.get_agent_context(agent_id, limit)
            return [
                {
                    "interaction_type": row['interaction_type'],
                    "data": row['data'],
                    "timestamp": row['created_at'].timestamp()
                }
                for row in agent_context
            ]
            
        except Exception as e:
            log_structured("agent_memory_retrieval_failed", error=str(e))
            return []
    
    async def _get_previous_interactions(self, agent_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get previous interactions with the target agent"""
        try:
            with self.database.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT interaction_type, data, created_at
                        FROM agent_interactions 
                        WHERE agent_id = %s 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (agent_id, limit))
                    
                    results = cur.fetchall()
                    
                    return [
                        {
                            "interaction_type": row['interaction_type'],
                            "data": row['data'],
                            "timestamp": row['created_at'].timestamp()
                        }
                        for row in results
                    ]
                    
        except Exception as e:
            log_structured("previous_interactions_retrieval_failed", error=str(e))
            return []
    
    async def _get_recent_interactions(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent interactions for preference analysis"""
        try:
            with self.database.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT agent_id, interaction_type, data, created_at
                        FROM agent_interactions 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (limit,))
                    
                    return cur.fetchall()
                    
        except Exception as e:
            log_structured("recent_interactions_retrieval_failed", error=str(e))
            return []

# Global context enricher instance
context_enricher = ContextEnricher()
