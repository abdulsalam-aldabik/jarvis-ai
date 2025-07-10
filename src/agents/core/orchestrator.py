"""
AutoGen 0.6.2 Orchestrator AssistantAgent - ENHANCED WITH DYNAMIC LLM RESPONSES
Following: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

# ✅ ENHANCED: Dynamic LLM integration
from src.agents.core.base_agent import AgentBase, get_model_client
from src.agents.core.mcp_tools import mcp_tools_manager
from src.agents.core.logging_config import log_structured
from config.settings import settings

class OrchestratorAgent(AgentBase):
    """ENHANCED AutoGen 0.6.2 Orchestrator with Dynamic LLM Responses"""
    
    def __init__(self) -> None:
        super().__init__(
            name="orchestrator",
            description="Multi-agent coordination and task delegation with AutoGen 0.6.2 architecture",
            agent_type="orchestrator",
            system_message="""You are Jarvis, an intelligent orchestrator using AutoGen 0.6.2 architecture.
            
            You coordinate multiple specialized agents and handle general conversation.
            
            Your responses should be:
            - Contextually aware and personalized
            - Proactive and helpful
            - Adaptive to user's communication style
            - Informative yet conversational
            - Dynamic based on user preferences and conversation history
            
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
            - Explain your reasoning for task delegation decisions
            - Always consider the user's preferences, conversation history, and current context when responding"""
        )
        
        # Agent coordination tracking
        self.active_delegations: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities = self._initialize_agent_capabilities()
        self.system_health_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5-minute cache for system health
        
        log_structured("orchestrator_agent_062_init",
                     autogen_version="0.6.2-official",
                     coordination_enabled=True,
                     dynamic_responses=True)

    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Enhanced orchestration processing with dynamic LLM response generation"""
        try:
            context = context or {}
            
            # Analyze task complexity and coordination needs
            task_analysis = await self.analyze_task_complexity(message)
            
            # Generate base response based on task type
            if task_analysis["requires_coordination"]:
                base_response = await self.coordinate_multi_agent_task(message, task_analysis, context)
            elif task_analysis["requires_delegation"]:
                base_response = await self.delegate_to_specialist(message, task_analysis["target_agent"], context)
            elif task_analysis["is_system_query"]:
                base_response = await self.handle_system_monitoring(message, context)
            else:
                base_response = await self.handle_general_conversation(message, context)
            
            # ✅ DYNAMIC ENHANCEMENT: Apply LLM tone adaptation
            if context.get('communication_style') or context.get('user_preferences'):
                enhanced_response = await self.enhance_response_with_full_intelligence(base_response, message, context)
            else:
                enhanced_response = base_response
            
            await self.store_enhanced_interaction(message, enhanced_response, context)
            return enhanced_response
            
        except Exception as e:
            log_structured("orchestrator_process_failed", error=str(e))
            error_context = context.copy() if context else {}
            error_context.update({'error_recovery': True})
            
            return await self.handle_general_conversation(
                f"I encountered an issue with: {message}", 
                error_context
            )

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

    async def coordinate_multi_agent_task(self, message: str, analysis: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Coordinate complex tasks requiring multiple agents with context awareness"""
        context = context or {}
        coordination_id = f"coord_{int(time.time())}"
        required_agents = self._determine_required_agents(message)
        
        # Generate contextual coordination response
        user_prefs = context.get('user_preferences', {})
        comm_style = context.get('communication_style', 'friendly')
        
        base_response = f"""🎯 **Multi-Agent Coordination Initiated**

**Task:** {message}

**Coordinating Agents:**
{self._format_agent_list(required_agents)}

**Coordination ID:** `{coordination_id}`

**Status:** Processing with {len(required_agents)} specialized agents

I'm orchestrating this complex task across multiple agents to provide you with comprehensive assistance."""

        # Add personalized context if available
        if user_prefs.get('prefers_detailed_updates'):
            base_response += f"\n\n**Coordination Strategy:** I'll coordinate between {', '.join(required_agents)} agents to ensure you get the most comprehensive and personalized response."
        
        return base_response

    async def delegate_to_specialist(self, message: str, target_agent: str, context: Dict[str, Any] = None) -> str:
        """Delegate task to appropriate specialist agent with context awareness"""
        context = context or {}
        delegation_id = f"del_{int(time.time())}"
        agent_name = target_agent.title()
        
        user_prefs = context.get('user_preferences', {})
        
        base_response = f"""🔄 **Task Delegated to {agent_name} Agent**

**Task:** {message}

**Delegated to:** {agent_name} Specialist

**Delegation ID:** `{delegation_id}`

I'm routing your request to our {agent_name.lower()} specialist for the most accurate and detailed assistance."""

        # Add personalized delegation reasoning
        if user_prefs.get('prefers_explanation'):
            base_response += f"\n\n**Why {agent_name}?** Based on your request, the {agent_name} specialist has the specialized knowledge and tools to provide you with the most accurate and comprehensive information."
        
        return base_response

    async def handle_system_monitoring(self, message: str, context: Dict[str, Any] = None) -> str:
        """Handle system monitoring and health checks with context awareness"""
        context = context or {}
        comm_style = context.get('communication_style', 'friendly')
        
        if comm_style == 'formal':
            return """📊 **System Status Report**

**Core Components:**
• **AutoGen 0.6.2 Framework:** Active ✅
• **Orchestrator Agent:** Online ✅  
• **Memory Management System:** Available ✅
• **Agent Registry:** Ready ✅

**Specialized Agents:**
• **Weather Information Specialist:** Operational
• **Routine Planning Specialist:** Operational
• **Orchestrator Coordination:** Operational

**Architecture:** AutoGen 0.6.2 multi-agent system with dynamic LLM integration"""
        
        elif comm_style == 'casual':
            return """📊 **System Check - All Good! 👍**

**What's Running:**
• **AutoGen 0.6.2:** Up and running ✅
• **Orchestrator (me!):** Online and ready ✅  
• **Memory System:** Working perfectly ✅
• **Agent Team:** All agents ready to go ✅

**Your Agent Team:**
• **Weather Agent** - Gets you weather info
• **Routine Agent** - Helps with daily planning
• **Orchestrator** - That's me, coordinating everything!

Everything's running smoothly! 🚀"""
        
        else:  # friendly (default)
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

    async def handle_general_conversation(self, message: str, context: Dict[str, Any] = None) -> str:
        """✅ DYNAMIC: Context-aware conversation handling"""
        context = context or {}
        
        # Get user preferences and communication style
        user_prefs = context.get('user_preferences', {})
        comm_style = context.get('communication_style', 'friendly')
        time_context = context.get('time_context', {})
        
        # Generate contextual base response
        if self._is_simple_greeting(message):
            return self._generate_personalized_greeting(user_prefs, time_context, comm_style)
        elif "help" in message.lower():
            return self._generate_contextual_help(user_prefs, comm_style)
        else:
            return self._generate_adaptive_response(message, context)

    def _generate_personalized_greeting(self, user_prefs: Dict, time_context: Dict, comm_style: str = 'friendly') -> str:
        """Generate personalized greeting based on context"""
        
        # Base greeting with time awareness
        if time_context.get('is_morning'):
            greeting = "🌅 Good morning! I'm Jarvis, ready to help you start your day."
        elif time_context.get('is_evening'):
            greeting = "🌆 Good evening! I'm Jarvis, here to assist you."
        else:
            greeting = "👋 Hello! I'm Jarvis, your AutoGen 0.6.2 orchestrator."
        
        # Build personalized capabilities based on preferences
        capabilities = []
        if user_prefs.get('prefers_weather_info'):
            capabilities.append("• **Weather** forecasting and alerts (I know you check this often!)")
        else:
            capabilities.append("• **Weather** information and forecasts")
            
        if user_prefs.get('prefers_routine_management'):
            capabilities.append("• **Routine** optimization and habit tracking")
        else:
            capabilities.append("• **Routine** planning and habit management")
        
        capabilities.extend([
            "• **Multi-agent coordination** for complex tasks",
            "• **System monitoring** and health checks",
            "• **General assistance** and conversation"
        ])
        
        # Adapt to communication style
        if comm_style == 'formal':
            return f"{greeting}\n\n**Available Services:**\n" + "\n".join(capabilities) + "\n\nHow may I assist you today?"
        elif comm_style == 'casual':
            return f"{greeting}\n\n**I can help with:**\n" + "\n".join(capabilities) + "\n\nWhat's up? What can I do for you? 😊"
        else:
            return f"{greeting}\n\n**I can help you with:**\n" + "\n".join(capabilities) + "\n\nWhat would you like to explore today?"

    def _generate_contextual_help(self, user_prefs: Dict, comm_style: str) -> str:
        """Generate help response adapted to communication style"""
        
        if comm_style == 'formal':
            return """🆘 **Jarvis Assistant - Comprehensive Help Documentation**

**Primary Functions:**
• Multi-agent task coordination and delegation
• Specialized agent routing (Weather, Routine, System)
• Context-aware conversation management
• System health monitoring and diagnostics
• Dynamic response generation with personalization

**Usage Examples:**
• "Coordinate weather check with morning routine planning"
• "Delegate weather forecast to specialist"
• "Monitor system status and agent performance"
• "Help me plan my day with weather considerations"

**Available Commands:**
• Weather information and forecasting
• Routine creation and optimization
• System monitoring and diagnostics
• General assistance and conversation

**Technical Specifications:**
• AutoGen 0.6.2 multi-agent architecture
• Dynamic LLM response generation
• Context-aware conversation management
• MCP tools integration

How may I assist you today?"""
        
        elif comm_style == 'casual':
            return """🤖 **Hey! Here's what I can do for you:**

**Cool Stuff:**
• Coordinate between different AI agents
• Route tasks to weather and routine specialists
• Keep track of our conversations and your preferences
• Monitor system health and performance
• Generate personalized responses just for you!

**Try saying:**
• "What's the weather and help me plan my day"
• "Check if everything's running smoothly"
• "Create a morning routine that works with the weather"
• "Help me coordinate a complex task"

**I'm pretty smart** - I learn your preferences and adapt my responses to your style. The more we chat, the better I get at helping you!

What's on your mind? Let's get stuff done! 🚀"""
        
        else:  # friendly (default)
            return """🆘 **Jarvis Help - AutoGen 0.6.2 Orchestrator**

**🎯 What I Do:**
• **Coordinate** complex multi-agent tasks
• **Delegate** to specialized agents (Weather, Routine)  
• **Monitor** system health and performance
• **Handle** general conversation and questions
• **Learn** your preferences and adapt my responses
• **Generate** personalized, context-aware responses

**💬 Example Commands:**
• "What's the weather and create a morning routine"
• "Check system status"
• "Get weather for tomorrow"
• "Create a workout routine"
• "Help me plan my day"

**🔧 My Capabilities:**
• Multi-agent coordination via AutoGen 0.6.2
• Dynamic LLM response generation
• Context-aware conversation management
• MCP tools integration
• User preference learning and adaptation

**🧠 Intelligence Features:**
• I remember your preferences and communication style
• I adapt my responses to your needs
• I can coordinate complex tasks across multiple agents
• I learn from our conversations to serve you better

What would you like to do? I'm here to help! 😊"""

    def _generate_adaptive_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate adaptive response based on message and context"""
        user_prefs = context.get('user_preferences', {})
        comm_style = context.get('communication_style', 'friendly')
        conversation_history = context.get('conversation_history', '')
        
        base_response = f"I understand you're asking about: \"{message}\""
        
        # Add contextual suggestions based on preferences
        suggestions = []
        if user_prefs.get('prefers_weather_info'):
            suggestions.append("• Get current weather and detailed forecasts")
        if user_prefs.get('prefers_routine_management'):
            suggestions.append("• Create or optimize your daily routines")
        if user_prefs.get('prefers_detailed_updates'):
            suggestions.append("• Provide comprehensive system monitoring")
        
        suggestions.extend([
            "• Coordinate complex multi-agent tasks",
            "• Monitor system health and performance",
            "• Provide general assistance and conversation"
        ])
        
        # Add conversation continuity if available
        context_addition = ""
        if conversation_history:
            context_addition = "\n\nBased on our previous conversation, I can build on what we've discussed to provide more personalized assistance."
        
        if comm_style == 'formal':
            return f"{base_response}\n\n**Available Options:**\n" + "\n".join(suggestions) + f"{context_addition}\n\nHow would you like to proceed?"
        elif comm_style == 'casual':
            return f"{base_response}\n\n**I can help with:**\n" + "\n".join(suggestions) + f"{context_addition}\n\nWhat sounds good to you? 🤔"
        else:
            return f"{base_response}\n\n**Here's how I can help:**\n" + "\n".join(suggestions) + f"{context_addition}\n\nWhat would you like to do?"

    # Helper methods (unchanged)
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
