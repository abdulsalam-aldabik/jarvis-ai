"""
AutoGen 0.6.2 AssistantAgent Base - Official Implementation
Following: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
"""
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# CORRECT AutoGen 0.6.2 AgentChat imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.memory import ListMemory
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

from config.settings import settings
from src.agents.core.database import db_manager
from src.agents.core.logging_config import log_structured

@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    content: str
    sender: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

def get_model_client() -> OllamaChatCompletionClient:
    """Get Ollama model client following official AutoGen 0.6.2 patterns"""
    return OllamaChatCompletionClient(
        model=settings.llm.default_model,
        base_url=f"{settings.llm.ollama_url}",
    )

class AgentBase(AssistantAgent):
    """CORRECT AutoGen 0.6.2 AssistantAgent base following official documentation"""
    
    def __init__(self, name: str, description: str, agent_type: str = None, system_message: str = None):
        # Create memory instance with a proper name
        list_memory = ListMemory(name=f"{name}_short_term")
        
        # ✅ FIXED: AutoGen 0.6.2 AssistantAgent initialization with LIST of memory objects
        super().__init__(
            name=name,
            description=description,
            model_client=get_model_client(),
            system_message=system_message or f"You are {name}, a {description}. You are helpful, knowledgeable, and maintain a conversational tone.",
            memory=[list_memory]  # ✅ CRITICAL FIX: List of Memory objects, not single object
        )
        
        self.agent_id = name
        self.agent_type = agent_type or self.__class__.__name__.lower()
        self.database = db_manager
        self.current_session_id = None
        
        # Add long-term vector memory with error handling
        try:
            self.vector_memory = ChromaDBVectorMemory(
                config=PersistentChromaDBVectorMemoryConfig(
                    collection_name=f"agent_{self.agent_id}_memory",
                    persistence_path=f"./memory_bank/{self.agent_id}",
                    k=3,  # Return top 3 results
                    score_threshold=0.4,  # Minimum similarity score
                    embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                        model_name="all-MiniLM-L6-v2"
                    ),
                )
            )
        except Exception as e:
            log_structured("vector_memory_init_failed", agent_id=self.agent_id, error=str(e))
            self.vector_memory = None
        
        # Register agent
        agentregistry.register_agent(self)
        log_structured("autogen_062_assistant_agent_initialized", 
                     agent_id=self.agent_id, 
                     agent_type=self.agent_type,
                     framework="autogen-0.6.2-official",
                     memory_count=len(self._memory))

    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Process message using AutoGen 0.6.2 AssistantAgent patterns"""
        try:
            # Create TextMessage following official patterns
            text_message = TextMessage(content=message, source="user")
            
            # Process with AssistantAgent
            result = await self.on_messages([text_message], context or {})
            
            # Store interaction
            await self.store_interaction(message, result.messages[-1].content)
            
            return result.messages[-1].content
            
        except Exception as e:
            error_msg = f"I encountered an error processing your request: {str(e)}"
            log_structured("process_message_failed", agent_id=self.agent_id, error=str(e))
            return error_msg

    def set_session_context(self, session_id: str):
        """Set session context for memory operations"""
        self.current_session_id = session_id
        log_structured("agent_session_context_set", 
                     agent_id=self.agent_id, 
                     session_id=session_id)

    async def store_interaction(self, user_input: str, response: str):
        """Store interaction in database and vector memory"""
        try:
            # Database storage
            interaction_id = self.database.store_agent_interaction(
                agent_id=self.agent_id,
                interaction_type="chat",
                data={
                    "user_input": user_input,
                    "agent_response": response,
                    "timestamp": time.time(),
                    "autogen_version": "0.6.2-official",
                }

            )
            
            # Vector memory storage (if available)
            if self.vector_memory and self.current_session_id:
                from autogen_core.memory import MemoryContent, MemoryMimeType
                
                interaction_content = f"User: {user_input}\n{self.agent_id}: {response}"
                await self.vector_memory.add(
                    MemoryContent(
                        content=interaction_content,
                        mime_type=MemoryMimeType.TEXT,
                        metadata={
                            "type": "agent_interaction",
                            "agent_id": self.agent_id,
                            "session_id": self.current_session_id,
                            "timestamp": time.time()
                        }
                    )
                )
                
        except Exception as e:
            log_structured("interaction_storage_failed", error=str(e))

class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
    
    def register_agent(self, agent: AgentBase) -> bool:
        try:
            agent_key = getattr(agent, 'agent_id', None) or getattr(agent, 'name', 'unknown')
            agent_type = getattr(agent, 'agent_type', 'unknown')
            self.agents[agent_key] = agent
            
            log_structured("autogen_062_assistant_agent_registered",
                         agent_name=agent_key,
                         agent_type=agent_type,
                         autogen_version="0.6.2-official")
            return True
            
        except Exception as e:
            log_structured("agent_registration_failed", 
                         agent_name=getattr(agent, 'name', 'unknown'),
                         error=str(e))
            return False

# Global agent registry
agentregistry = AgentRegistry()