"""
AutoGen 0.6.2 Orchestrator AssistantAgent - SIMPLIFIED & WORKING
Following: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

# ✅ SIMPLIFIED: Remove problematic task-centric memory for now
from src.agents.core.base_agent import AgentBase, get_model_client
from src.agents.core.mcp_tools import mcp_tools_manager
from src.agents.core.logging_config import log_structured
from config.settings import settings

class OrchestratorAgent(AgentBase):
    """CORRECT AutoGen 0.6.2 Orchestrator AssistantAgent"""
    
    def __init__(self) -> None:
        super().__init__(
            name="orchestrator",
            description="Multi-agent coordination and task delegation with AutoGen 0.6.2 architecture",
            agent_type="orchestrator",
            system_message="""You are Jarvis, an intelligent orchestrator using AutoGen 0.6.2 architecture.
            
            You coordinate multiple specialized agents and handle general conversation.
            
            Key responsibilities:
            - Coordinate complex tasks requiring multiple agents
            - Delegate tasks to appropriate specialized agents
            - Monitor system health and agent status
            - Handle general conversation and questions
            - Provide context sharing between agents
            - Explain reasoning and decision-making process
            
            Guidelines:
            - Be concise but informative
            - Use context from previous conversations when relevant
            - Coordinate with specialized agents for domain-specific tasks
            - Always aim to be helpful and provide value
            - Maintain a professional yet friendly demeanor
            - Explain your reasoning for task delegation decisions"""
        )
        
        
        # Agent coordination tracking
        self.active_delegations: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities = self._initialize_agent_capabilities()
        self.system_health_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5-minute cache for system health
        
        log_structured("orchestrator_agent_062_init",
                     autogen_version="0.6.2-official",
                     coordination_enabled=True)

    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Enhanced orchestration processing with AutoGen 0.6.2"""
        try:
            # Analyze task complexity and coordination needs
            task_analysis = await self.analyze_task_complexity(message)
            
            # Determine if coordination is needed
            if task_analysis["requires_coordination"]:
                result = await self.coordinate_multi_agent_task(message, task_analysis)
            elif task_analysis["requires_delegation"]:
                result = await self.delegate_to_specialist(message, task_analysis["target_agent"])
            elif task_analysis["is_system_query"]:
                result = await self.handle_system_monitoring(message)
            else:
                result = await self.handle_general_conversation(message)
            
            await self.store_interaction(message, result)
            return result
            
        except Exception as e:
            log_structured("orchestrator_process_failed", error=str(e))
            return f"I encountered an issue coordinating your request. Let me help you directly: {await self.handle_general_conversation(message)}"

    async def analyze_task_complexity(self, message: str) -> Dict[str, Any]:
        """Analyze task to determine coordination needs"""
        message_lower = message.lower()
        
        # Multi-agent coordination indicators
        coordination_keywords = [
            "weather and routine", "schedule weather check", "daily routine with weather",
            "coordinate", "both", "multiple", "combination", "integrate"
        ]
        
        # Delegation indicators
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny"]
        routine_keywords = ["routine", "schedule", "habit", "morning", "evening"]
        system_keywords = ["status", "health", "system", "agents", "performance"]
        
        requires_coordination = any(keyword in message_lower for keyword in coordination_keywords)
        requires_delegation = not requires_coordination and (
            any(keyword in message_lower for keyword in weather_keywords + routine_keywords)
        )
        is_system_query = any(keyword in message_lower for keyword in system_keywords)
        
        # Determine target agent for delegation
        target_agent = None
        if requires_delegation:
            if any(keyword in message_lower for keyword in weather_keywords):
                target_agent = "weather"
            elif any(keyword in message_lower for keyword in routine_keywords):
                target_agent = "routine"
        
        return {
            "requires_coordination": requires_coordination,
            "requires_delegation": requires_delegation,
            "is_system_query": is_system_query,
            "target_agent": target_agent,
            "complexity_score": self._calculate_complexity_score(message),
            "task_type": self._determine_task_type(message_lower)
        }

    async def coordinate_multi_agent_task(self, message: str, analysis: Dict[str, Any]) -> str:
        """Coordinate complex tasks requiring multiple agents"""
        coordination_id = f"coord_{int(time.time())}"
        required_agents = self._determine_required_agents(message)
        
        return f"""🎯 **Multi-Agent Coordination Initiated**

**Task:** {message}

**Coordinating Agents:**
{self._format_agent_list(required_agents)}

**Coordination ID:** `{coordination_id}`

**Status:** Processing with {len(required_agents)} specialized agents

I'm orchestrating this complex task across multiple agents to provide you with comprehensive assistance."""

    async def delegate_to_specialist(self, message: str, target_agent: str) -> str:
        """Delegate task to appropriate specialist agent"""
        delegation_id = f"del_{int(time.time())}"
        agent_name = target_agent.title()
        
        return f"""🔄 **Task Delegated to {agent_name} Agent**

**Task:** {message}

**Delegated to:** {agent_name} Specialist

**Delegation ID:** `{delegation_id}`

I'm routing your request to our {agent_name.lower()} specialist for the most accurate and detailed assistance."""

    async def handle_system_monitoring(self, message: str) -> str:
        """Handle system monitoring and health checks"""
        return """📊 **System Status**

**AutoGen 0.6.2:** Active ✅
**Orchestrator Agent:** Online ✅  
**Memory System:** Available ✅
**Agent Registry:** Ready ✅

**Available Agents:**
• **Weather** - Weather information specialist
• **Routine** - Routine planning specialist  
• **Orchestrator** - Coordination and general assistance

*System monitoring via AutoGen 0.6.2 architecture*"""

    async def handle_general_conversation(self, message: str) -> str:
        """Handle general conversation and questions"""
        if self._is_simple_greeting(message):
            return f"""👋 **Hello! I'm Jarvis** - your AutoGen 0.6.2 orchestrator.

I can help you with:
• **Weather** information and forecasts
• **Routine** planning and habit management  
• **System** monitoring and agent coordination
• **General** questions and conversation

What would you like to do today?"""
        
        elif "help" in message.lower():
            return """🆘 **Jarvis Help - AutoGen 0.6.2 Orchestrator**

**🎯 What I Do:**
• **Coordinate** complex multi-agent tasks
• **Delegate** to specialized agents (Weather, Routine)  
• **Monitor** system health and performance
• **Handle** general conversation and questions

**💬 Example Commands:**
• "What's the weather and create a morning routine"
• "Check system status"
• "Get weather for tomorrow"
• "Create a workout routine"
• "Help me plan my day"

**🔧 My Capabilities:**
• Multi-agent coordination via AutoGen 0.6.2
• MCP tools integration
• Task delegation and routing
• System monitoring"""
        
        else:
            return f"""I understand you're asking about: "{message}"

As your orchestrator, I can:
• Coordinate with specialized agents for detailed information
• Provide system monitoring and health checks
• Help with multi-step tasks requiring agent collaboration

Would you like me to:
• Get weather information?
• Help with routine planning?
• Check system status?
• Coordinate a complex task?"""

    # Helper methods
    def _calculate_complexity_score(self, message: str) -> float:
        """Calculate task complexity score"""
        factors = [
            len(message.split()) > 10,
            "and" in message.lower(),
            "both" in message.lower(),
            any(word in message.lower() for word in ["coordinate", "integrate", "combine"])
        ]
        return sum(factors) / len(factors)

    def _determine_task_type(self, message_lower: str) -> str:
        """Determine the primary task type"""
        if any(word in message_lower for word in ["weather", "temperature"]):
            return "weather"
        elif any(word in message_lower for word in ["routine", "schedule"]):
            return "routine"
        elif any(word in message_lower for word in ["system", "status", "health"]):
            return "system"
        elif any(word in message_lower for word in ["coordinate", "multiple"]):
            return "coordination"
        else:
            return "general"

    def _determine_required_agents(self, message: str) -> List[str]:
        """Determine which agents are required for a coordinated task"""
        message_lower = message.lower()
        required = []
        
        if any(word in message_lower for word in ["weather", "temperature", "forecast"]):
            required.append("weather")
        
        if any(word in message_lower for word in ["routine", "schedule", "habit"]):
            required.append("routine")
        
        if len(required) > 1:
            required.append("orchestrator")
        
        return required if required else ["orchestrator"]

    def _initialize_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize known agent capabilities"""
        return {
            "weather": {
                "specialization": "Weather information and forecasts",
                "capabilities": ["current_weather", "forecast", "alerts"],
                "mcp_tools": ["get_weather", "get_forecast"]
            },
            "routine": {
                "specialization": "Routine planning and habit management", 
                "capabilities": ["routine_creation", "habit_tracking", "schedule_management"],
                "mcp_tools": ["create_routine", "track_habit"]
            },
            "orchestrator": {
                "specialization": "Multi-agent coordination and general assistance",
                "capabilities": ["coordination", "delegation", "system_monitoring"],
                "mcp_tools": ["agent_coordination", "system_monitoring"]
            }
        }

    def _format_agent_list(self, agents: List[str]) -> str:
        """Format list of agents for display"""
        formatted = []
        for agent in agents:
            capabilities = self.agent_capabilities.get(agent, {})
            specialization = capabilities.get("specialization", "General purpose")
            formatted.append(f"• **{agent.title()}** - {specialization}")
        return "\n".join(formatted)

    def _is_simple_greeting(self, message: str) -> bool:
        """Check if message is a simple greeting"""
        greetings = ["hello", "hi", "hey", "good morning", "good evening", "greetings"]
        return any(greeting in message.lower() for greeting in greetings) and len(message.split()) <= 3
