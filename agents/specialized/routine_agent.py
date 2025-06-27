"""
Routine agent compatible with corrected AutoGen 2.x patterns
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from autogen_core import MessageContext
from agents.core.base_agent import AutoGenBaseAgent
from agents.core.database import db_manager
from agents.core.logging_config import log_structured
from config.settings import settings

class ReliableRoutineAgent(AutoGenBaseAgent):
    """Reliable AutoGen routine management agent with session awareness"""
    
    def __init__(self):
        super().__init__(
            name="routine",
            description="Routine planning and management with session context",
            agent_type="routine"
        )

    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process routine-related messages with session context"""
        try:
            log_structured("routine_processing", message=message[:100], 
                         sender=str(ctx.sender), agent_id=self.agent_id,
                         session_id=self.current_session_id)
            
            message_lower = message.lower()
            
            if any(word in message_lower for word in ["create", "make", "new"]):
                response = await self.create_routine(message)
            elif any(word in message_lower for word in ["list", "show", "my routines"]):
                response = await self.list_routines()
            elif any(word in message_lower for word in ["morning", "evening", "daily"]):
                response = await self.suggest_routine_type(message)
            else:
                response = self.provide_routine_help()
            
            # Store interaction with session context
            await self._store_interaction(message, response)
            return response
            
        except Exception as e:
            log_structured("routine_error", error=str(e), message=message[:100], 
                         agent_id=self.agent_id, session_id=self.current_session_id)
            return "I'm having trouble with routine planning right now. Please try again."

    async def create_routine(self, message: str) -> str:
        """Create a new routine based on user request"""
        try:
            # Simple routine creation logic
            routine_type = "general"
            if "morning" in message.lower():
                routine_type = "morning"
            elif "evening" in message.lower():
                routine_type = "evening"
            elif "workout" in message.lower():
                routine_type = "workout"
            
            routine_data = {
                "name": f"{routine_type}_routine_{int(time.time())}",
                "description": f"Auto-generated {routine_type} routine",
                "routine_type": routine_type,
                "activities": self.get_default_activities(routine_type),
                "created_from": message,
                "session_id": self.current_session_id
            }
            
            # Store routine in database
            interaction_id = db_manager.store_agent_interaction(
                agent_id=self.agent_id,
                interaction_type="routine_created",
                data=routine_data
            )
            
            activities = ", ".join(routine_data["activities"])
            
            log_structured("routine_created", routine_type=routine_type, 
                         interaction_id=interaction_id, agent_id=self.agent_id,
                         session_id=self.current_session_id)
            
            return f"I've created a {routine_type} routine for you with these activities: {activities}. Would you like me to customize it further?"
            
        except Exception as e:
            log_structured("routine_creation_failed", error=str(e), agent_id=self.agent_id)
            return f"I couldn't create the routine: {str(e)[:50]}"

    async def list_routines(self) -> str:
        """List existing routines for the current session"""
        try:
            context = db_manager.get_agent_context(self.agent_id, limit=5)
            routine_interactions = [ctx for ctx in context if ctx.get("interaction_type") == "routine_created"]
            
            if not routine_interactions:
                return "You don't have any routines yet. Would you like me to create one for you?"
            
            routines = []
            for interaction in routine_interactions[:3]:  # Limit to 3 most recent
                data = interaction.get("data", {})
                routines.append(f"• {data.get('routine_type', 'General')} routine")
            
            return f"Your routines:\n" + "\n".join(routines)
            
        except Exception as e:
            log_structured("routine_list_failed", error=str(e), agent_id=self.agent_id)
            return "I couldn't retrieve your routines right now."

    async def suggest_routine_type(self, message: str) -> str:
        """Suggest routine based on type mentioned"""
        message_lower = message.lower()
        
        if "morning" in message_lower:
            return """Here's a great morning routine structure:
• Wake up at consistent time
• Light stretching or exercise
• Healthy breakfast
• Review daily goals
• Meditation or mindfulness (5-10 minutes)

Would you like me to create this routine for you?"""
        
        elif "evening" in message_lower:
            return """Here's a relaxing evening routine:
• Wind down activities (reading, light music)
• Prepare for tomorrow (clothes, tasks)
• Reflection or journaling
• Limit screen time 1 hour before bed
• Consistent sleep time

Shall I set this up as your evening routine?"""
        
        else:
            return "I can help you create morning, evening, workout, or custom routines. What type interests you?"

    def get_default_activities(self, routine_type: str) -> List[str]:
        """Get default activities for routine types"""
        activities = {
            "morning": ["Wake up", "Exercise", "Breakfast", "Plan day"],
            "evening": ["Wind down", "Prepare tomorrow", "Read", "Sleep"],
            "workout": ["Warm up", "Exercise", "Cool down", "Hydrate"],
            "general": ["Start activity", "Complete task", "Review progress"]
        }
        return activities.get(routine_type, activities["general"])

    def provide_routine_help(self) -> str:
        """Provide general routine help"""
        return """I can help you with:
• Creating morning, evening, or workout routines
• Listing your existing routines
• Suggesting routine improvements
• Tracking routine progress

Try asking "Create a morning routine" or "Show my routines"!"""
