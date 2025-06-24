import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from autogen_core import RoutedAgent, MessageContext, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from dataclasses import dataclass
from config.settings import settings
from agents.core.database import db_manager

@dataclass
class AgentTask:
    """Simple task structure for AutoGen messaging"""
    content: str
    context: Dict[str, Any]
    sender: str

@dataclass
class AgentResponse:
    """Simple response structure"""
    content: str
    success: bool
    metadata: Dict[str, Any]

class AutoGenBaseAgent(RoutedAgent):
    """AutoGen-native base agent using RoutedAgent"""
    
    def __init__(self, agent_type: str, description: str = ""):
        super().__init__(description=description or f"{agent_type} agent using AutoGen")
        self.agent_type = agent_type
        self.agent_id = f"{agent_type}_agent"
        
        # Register in database
        db_manager.register_agent(
            agent_id=self.agent_id,
            agent_type=agent_type,
            capabilities={"description": description}
        )
        
        # Update heartbeat
        db_manager.update_agent_heartbeat(self.agent_id)
    
    @message_handler
    async def handle_user_message(self, message: str, ctx: MessageContext) -> str:
        """AutoGen native message handler"""
        try:
            # Log the interaction
            db_manager.log_event("INFO", f"Processing: {message[:100]}", 
                               {"sender": str(ctx.sender)}, self.agent_id)
            
            # Process using agent-specific logic
            response = await self.process_message(message, ctx)
            
            # Store in semantic memory
            from learning.behavior.behavior_engine import add_to_semantic_memory
            add_to_semantic_memory(
                f"User: {message}\n{self.agent_type}: {response}",
                {"type": "conversation", "agent": self.agent_type, "timestamp": time.time()}
            )
            
            return response
            
        except Exception as e:
            db_manager.log_event("ERROR", f"Message handling failed: {str(e)}", 
                               {"message": message[:100]}, self.agent_id)
            return f"I'm having trouble processing that request: {str(e)}"
    
    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Override this in specialized agents"""
        return f"I'm a {self.agent_type} agent. I received: {message}"
