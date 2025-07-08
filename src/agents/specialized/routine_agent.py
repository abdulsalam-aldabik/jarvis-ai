"""
AutoGen 0.6.2 Routine AssistantAgent - CORRECTED Memory Access
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, time as dt_time

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

# ✅ SIMPLIFIED: Remove task-centric memory for now to avoid complications
from src.agents.core.base_agent import AgentBase, get_model_client
from src.agents.core.mcp_tools import mcp_tools_manager
from src.agents.core.logging_config import log_structured
from config.settings import settings

class RoutineAgent(AgentBase):
    """CORRECT AutoGen 0.6.2 Routine AssistantAgent - SIMPLIFIED"""
    
    def __init__(self):
        super().__init__(
            name="routine",
            description="Routine planning and habit management with AutoGen 0.6.2",
            agent_type="routine",
            system_message="""You are a routine and habit management specialist using AutoGen 0.6.2 architecture.
            
            You help users create, manage, and track daily routines with intelligent suggestions.
            
            Key capabilities:
            - Create personalized morning, evening, and workout routines
            - Track routine execution and suggest improvements
            - Provide contextual routine recommendations
            - Manage routine schedules and reminders
            
            Guidelines:
            - Be encouraging and supportive about routine building
            - Suggest realistic, achievable routine steps
            - Focus on habit formation and consistency
            - Provide clear, actionable routine structures"""
        )
        
        # ✅ REMOVED: Problematic memory controller and teachability
        # Keep it simple for now until the basic system works
        
        # Routine tracking
        self.routine_cache: Dict[str, Dict[str, Any]] = {}
        self.default_routines = self._initialize_default_routines()
        
        log_structured("routine_agent_062_init",
                     autogen_version="0.6.2-official",
                     mcp_enabled=True)

    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Enhanced routine processing with AutoGen 0.6.2"""
        try:
            if not self.is_routine_query(message):
                return await self.get_help_response()
            
            # Analyze routine intent
            intent = self.analyze_routine_intent(message)
            
            # Process based on intent - simplified for now
            if intent == "create":
                result = await self.create_routine_via_mcp(message)
            elif intent == "list":
                result = await self.list_routines_via_mcp()
            elif intent == "suggest":
                result = await self.suggest_routine(message)
            elif intent == "track":
                result = await self.track_routine_execution(message)
            else:
                result = await self.get_help_response()
            
            await self.store_interaction(message, result)
            return result
            
        except Exception as e:
            log_structured("routine_process_failed", error=str(e))
            return f"I'm having trouble with routine planning right now. Please try again."

    async def create_routine_via_mcp(self, message: str) -> str:
        """Create routine using MCP tools - simplified"""
        try:
            routine_type = self.extract_routine_type(message)
            base_routine = self.default_routines.get(routine_type, self.default_routines["general"])
            
            # Try MCP tools
            tools = await mcp_tools_manager.get_available_tools()
            routine_tool = None
            
            for tool in tools:
                if "routine" in tool.name.lower() and "create" in tool.name.lower():
                    routine_tool = tool
                    break
            
            routine_data = {
                "name": f"{routine_type.title()} Routine",
                "type": routine_type,
                "activities": base_routine["activities"],
                "duration_minutes": base_routine.get("duration", 30),
                "created_from": message,
                "session_id": self.current_session_id
            }
            
            if routine_tool:
                await mcp_tools_manager.call_mcp_tool(
                    routine_tool.name,
                    {"routine_data": routine_data}
                )
            else:
                # Fallback to database storage
                self.database.store_agent_interaction(
                    agent_id=self.agent_id,
                    interaction_type="routine_created",
                    data=routine_data
                )
            
            activities_list = "\n".join([f"• {activity}" for activity in base_routine["activities"]])
            
            return f"""✅ Created your **{routine_type} routine**!

**Activities:**
{activities_list}

**Duration:** ~{base_routine.get('duration', 30)} minutes

Would you like me to customize it further or set up reminders?"""
            
        except Exception as e:
            log_structured("routine_creation_mcp_failed", error=str(e))
            return f"I couldn't create the routine: {str(e)[:50]}"

    # ✅ ADD: All the missing helper methods
    def analyze_routine_intent(self, message: str) -> str:
        """Analyze user intent for routine management"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["create", "make", "new", "build"]):
            return "create"
        elif any(word in message_lower for word in ["list", "show", "my routines", "what routines"]):
            return "list"
        elif any(word in message_lower for word in ["suggest", "recommend", "ideas", "help me"]):
            return "suggest"
        elif any(word in message_lower for word in ["completed", "done", "finished", "track"]):
            return "track"
        else:
            return "general"

    def extract_routine_type(self, message: str) -> str:
        """Extract routine type from message"""
        message_lower = message.lower()
        
        if "morning" in message_lower:
            return "morning"
        elif "evening" in message_lower or "night" in message_lower:
            return "evening"
        elif "workout" in message_lower or "exercise" in message_lower:
            return "workout"
        else:
            return "general"

    def is_routine_query(self, message: str) -> bool:
        """Check if message is routine-related"""
        keywords = ["routine", "habit", "schedule", "morning", "evening", "workout", "daily"]
        return any(keyword in message.lower() for keyword in keywords)

    def _initialize_default_routines(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default routine templates"""
        return {
            "morning": {
                "activities": [
                    "Wake up at consistent time",
                    "Drink a glass of water",
                    "Light stretching (5-10 minutes)",
                    "Healthy breakfast",
                    "Review daily goals",
                    "Brief meditation or mindfulness"
                ],
                "duration": 45
            },
            "evening": {
                "activities": [
                    "Wind down activities (reading, music)",
                    "Prepare for tomorrow (clothes, tasks)",
                    "Reflection or journaling",
                    "Limit screen time (1 hour before bed)",
                    "Consistent sleep time"
                ],
                "duration": 30
            },
            "workout": {
                "activities": [
                    "Warm up (5-10 minutes)",
                    "Main exercise routine",
                    "Cool down and stretching",
                    "Hydration",
                    "Log workout progress"
                ],
                "duration": 60
            },
            "general": {
                "activities": [
                    "Start focused activity",
                    "Complete main task",
                    "Take breaks as needed",
                    "Review progress"
                ],
                "duration": 30
            }
        }

    async def list_routines_via_mcp(self) -> str:
        """List routines using MCP tools"""
        try:
            tools = await mcp_tools_manager.get_available_tools()
            list_tool = None
            
            for tool in tools:
                if "routine" in tool.name.lower() and ("list" in tool.name.lower() or "get" in tool.name.lower()):
                    list_tool = tool
                    break
            
            if list_tool:
                result = await mcp_tools_manager.call_mcp_tool(
                    list_tool.name,
                    {"session_id": self.current_session_id}
                )
                return self.format_routine_list(result)
            else:
                return "You don't have any routines yet. Would you like me to create one for you?"
                
        except Exception as e:
            log_structured("routine_list_failed", error=str(e))
            return "I couldn't retrieve your routines right now."

    async def suggest_routine(self, message: str) -> str:
        """Suggest routine"""
        routine_type = self.extract_routine_type(message)
        return self.get_base_suggestion(routine_type) + "\n\nShall I create this routine for you?"

    async def track_routine_execution(self, message: str) -> str:
        """Track routine execution and provide feedback"""
        completed = any(word in message.lower() for word in ["completed", "done", "finished"])
        
        if completed:
            return """🎉 **Great job completing your routine!**

Consistency builds habits. Keep it up!

Would you like me to:
• Adjust your routine based on today's experience?
• Set up reminders for tomorrow?
• Track your progress over time?"""
        else:
            return """**Routine tracking helps build consistency!**

Let me know when you've completed activities by saying:
• "I completed my morning routine"
• "Finished my workout"
• "Done with evening routine"

I'll help track your progress and suggest improvements."""

    def get_base_suggestion(self, routine_type: str) -> str:
        """Get base routine suggestion"""
        suggestions = {
            "morning": """**Here's a great morning routine structure:**

🌅 **Morning Energizer Routine**
• Wake up at consistent time
• Hydrate with a glass of water
• Light stretching or movement (5-10 minutes)
• Healthy breakfast
• Review daily priorities
• 5 minutes meditation or mindfulness

*Duration: ~45 minutes*""",
            
            "evening": """**Here's a relaxing evening routine:**

🌙 **Evening Wind-Down Routine**
• Wind down activities (reading, soft music)
• Prepare for tomorrow (clothes, priorities)
• Reflection or gratitude journaling
• Digital sunset (screens off 1 hour before bed)
• Consistent sleep time

*Duration: ~30 minutes*""",
            
            "workout": """**Here's an effective workout routine structure:**

💪 **Balanced Workout Routine**
• Warm up (5-10 minutes)
• Strength or cardio training
• Cool down and stretching
• Hydration break
• Log progress and celebrate!

*Duration: ~60 minutes*""",
            
            "general": """**Here's a flexible routine framework:**

📋 **Productive Daily Routine**
• Set clear intention
• Focus on priority task
• Take purposeful breaks
• Review and adjust
• Celebrate progress

*Duration: ~30 minutes*"""
        }
        
        return suggestions.get(routine_type, suggestions["general"])

    def format_routine_list(self, mcp_result: Any) -> str:
        """Format routine list from MCP tool result"""
        try:
            if isinstance(mcp_result, dict) and "routines" in mcp_result:
                routines = mcp_result["routines"]
                if not routines:
                    return "You don't have any routines yet. Would you like me to create one?"
                
                formatted = "**Your routines:**\n"
                for routine in routines[:5]:
                    name = routine.get("name", "Unnamed Routine")
                    activities_count = len(routine.get("activities", []))
                    formatted += f"• **{name}** - {activities_count} activities\n"
                
                return formatted + "\nSay 'create a routine' to add more!"
            else:
                return str(mcp_result)
                
        except Exception:
            return "You don't have any routines yet. Would you like me to create one?"

    async def get_help_response(self) -> str:
        """Generate help response"""
        return """**I can help you with routine management!** 

🎯 **What I can do:**
• **Create** personalized morning, evening, or workout routines
• **List** your existing routines
• **Suggest** routine improvements
• **Track** routine completion and progress

💬 **Try saying:**
• "Create a morning routine"
• "Show my routines"
• "Suggest a workout routine"
• "I completed my routine"

I'm here to help you build consistent, healthy habits!"""
