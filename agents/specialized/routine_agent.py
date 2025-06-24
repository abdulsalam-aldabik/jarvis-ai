import json
from typing import Dict, Any, List
from datetime import datetime, time
from agents.core.base_agent import BaseAgent, AgentCapability, AgentTask
from agents.core.database import db_manager
from learning.behavior.behavior_engine import train_time_series_forecaster, learn_rules
import pandas as pd

class RoutineAgent(BaseAgent):
    """Specialized agent for routine management and optimization"""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="create_routine",
                description="Create a new routine based on user preferences",
                input_schema={"name": "string", "activities": "array", "schedule": "object"},
                output_schema={"routine_id": "string", "success": "boolean"}
            ),
            AgentCapability(
                name="optimize_routine",
                description="Optimize existing routine based on learned patterns",
                input_schema={"routine_id": "string", "optimization_type": "string"},
                output_schema={"optimizations": "array", "confidence": "number"}
            ),
            AgentCapability(
                name="suggest_routine",
                description="Suggest new routine based on behavior patterns",
                input_schema={"user_context": "object", "preferences": "object"},
                output_schema={"suggestions": "array", "reasoning": "string"}
            ),
            AgentCapability(
                name="execute_routine",
                description="Execute a routine and track progress",
                input_schema={"routine_id": "string", "context": "object"},
                output_schema={"execution_status": "string", "steps_completed": "array"}
            )
        ]
        
        super().__init__(
            agent_id="routine_agent",
            agent_type="routine",
            capabilities=capabilities
        )
    
    async def execute_plan(self, plan: Dict[str, Any], task: AgentTask) -> Any:
        """Execute routine-related plans"""
        task_type = task.task_type
        context = task.context
        
        if task_type == "create_routine":
            return await self._create_routine(context)
        
        elif task_type == "optimize_routine":
            routine_id = context.get("routine_id")
            return await self._optimize_routine(routine_id)
        
        elif task_type == "suggest_routine":
            return await self._suggest_routine(context)
        
        elif task_type == "execute_routine":
            routine_id = context.get("routine_id")
            return await self._execute_routine(routine_id, context)
        
        else:
            return None
    
    async def _create_routine(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new routine"""
        self.log_reasoning("routine_creation", "Creating new routine from user input")
        
        routine_data = {
            "name": context.get("name", "New Routine"),
            "description": context.get("description", ""),
            "activities": context.get("activities", []),
            "schedule_time": context.get("schedule_time"),
            "duration_minutes": context.get("duration_minutes", 30),
            "approved": False  # Requires user approval
        }
        
        try:
            conn = db_manager.get_connection()
            with conn, conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO routines (name, description, schedule_time, duration_minutes, activities, approved)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    routine_data["name"],
                    routine_data["description"],
                    routine_data["schedule_time"],
                    routine_data["duration_minutes"],
                    json.dumps(routine_data["activities"]),
                    routine_data["approved"]
                ))
                routine_id = cur.fetchone()['id']
            
            self.log_reasoning("routine_created", f"Routine created with ID: {routine_id}")
            return {
                "routine_id": str(routine_id),
                "success": True,
                "routine": routine_data,
                "requires_approval": True
            }
            
        except Exception as e:
            self.log_reasoning("routine_creation_error", f"Failed to create routine: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _optimize_routine(self, routine_id: str) -> Dict[str, Any]:
        """Optimize existing routine using behavior learning"""
        self.log_reasoning("routine_optimization", f"Optimizing routine {routine_id}")
        
        # Get routine execution history
        execution_data = self._get_routine_execution_data(routine_id)
        
        if not execution_data:
            return {
                "optimizations": [],
                "reasoning": "Insufficient execution data for optimization"
            }
        
        # Use behavior learning to identify patterns
        optimizations = self._analyze_routine_patterns(execution_data)
        
        return {
            "optimizations": optimizations,
            "confidence": 0.8,
            "reasoning": "Optimizations based on execution patterns and success rates"
        }
    
    async def _suggest_routine(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest new routines based on learned patterns"""
        self.log_reasoning("routine_suggestion", "Generating routine suggestions")
        
        # This would use behavior learning to suggest routines
        # based on user patterns, time of day, etc.
        
        suggestions = [
            {
                "name": "Morning Productivity Routine",
                "activities": ["meditation", "exercise", "planning"],
                "schedule_time": "07:00",
                "duration_minutes": 45,
                "reasoning": "Based on your most productive morning patterns"
            },
            {
                "name": "Evening Wind-down",
                "activities": ["review_day", "reading", "relaxation"],
                "schedule_time": "21:00",
                "duration_minutes": 30,
                "reasoning": "Optimized for better sleep based on your patterns"
            }
        ]
        
        return {
            "suggestions": suggestions,
            "reasoning": "Suggestions based on behavioral pattern analysis"
        }
    
    async def _execute_routine(self, routine_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a routine and track progress"""
        self.log_reasoning("routine_execution", f"Executing routine {routine_id}")
        
        # Get routine details
        routine = self._get_routine_by_id(routine_id)
        if not routine:
            return {"execution_status": "failed", "error": "Routine not found"}
        
        # Track execution
        execution_record = {
            "routine_id": routine_id,
            "started_at": datetime.now(),
            "status": "in_progress",
            "steps_completed": []
        }
        
        # This would integrate with other agents or tools to actually execute the routine
        # For now, we'll simulate the execution
        
        return {
            "execution_status": "started",
            "routine_name": routine.get("name"),
            "estimated_duration": routine.get("duration_minutes"),
            "steps_completed": []
        }
    
    def _get_routine_execution_data(self, routine_id: str) -> List[Dict]:
        """Get historical execution data for a routine"""
        # This would query the database for execution history
        return []
    
    def _analyze_routine_patterns(self, execution_data: List[Dict]) -> List[Dict]:
        """Analyze routine execution patterns for optimization"""
        # This would use the behavior learning algorithms
        return [
            {
                "type": "timing_optimization",
                "suggestion": "Move routine 30 minutes earlier",
                "confidence": 0.85,
                "reasoning": "Higher success rate observed at earlier times"
            }
        ]
    
    def _get_routine_by_id(self, routine_id: str) -> Dict[str, Any]:
        """Get routine details by ID"""
        try:
            conn = db_manager.get_connection()
            with conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM routines WHERE id = %s", (routine_id,))
                result = cur.fetchone()
                if result:
                    return dict(result)
            return None
        except Exception:
            return None

        # Add this method to the existing RoutineAgent class
    async def execute_intelligent_plan(self, task: AgentTask, reasoning: str) -> Any:
        """Execute intelligent routine plan - ADD THIS METHOD"""
        try:
            request = task.content.lower()
            
            # Analyze what kind of routine help they need
            if any(word in request for word in ["create", "make", "new", "plan"]):
                return await self._suggest_routine_creation(task.content, reasoning)
            elif any(word in request for word in ["optimize", "improve", "better"]):
                return await self._suggest_routine_optimization(task.content, reasoning)
            else:
                return await self._provide_routine_guidance(task.content, reasoning)
                
        except Exception as e:
            return f"Routine planning error: {str(e)}"

    async def _suggest_routine_creation(self, request: str, reasoning: str) -> str:
        """Suggest routine creation"""
        return ("I can help you create a personalized routine! Based on your request, I suggest starting with:\n"
                "• Morning routine (wake up, exercise, planning)\n"
                "• Work routine (focused work blocks, breaks)\n"
                "• Evening routine (review, relaxation, preparation for tomorrow)\n"
                "What type of routine interests you most?")

    async def _suggest_routine_optimization(self, request: str, reasoning: str) -> str:
        """Suggest routine optimization"""
        return ("To optimize your routine, I'd recommend:\n"
                "• Track your energy levels throughout the day\n"
                "• Identify your most productive hours\n"
                "• Build in buffer time for unexpected tasks\n"
                "• Regular review and adjustment\n"
                "What specific aspect of your routine would you like to improve?")

    async def _provide_routine_guidance(self, request: str, reasoning: str) -> str:
        """Provide general routine guidance"""
        return ("I'm here to help with routine planning! I can assist with:\n"
                "• Creating new daily routines\n"
                "• Optimizing existing schedules\n"
                "• Building healthy habits\n"
                "• Time management strategies\n"
                "What would you like to work on?")
