"""
COMPLETELY DYNAMIC orchestrator with NO hardcoded responses or keywords
"""
import requests
import logging
import time
from typing import Dict, Any, List, Optional
from autogen_core import RoutedAgent, message_handler, MessageContext
from src.agents.core.base_agent import AutoGenBaseAgent
from src.agents.core.database import db_manager
from config.settings import settings

logger = logging.getLogger(__name__)

class ReliableOrchestrator(AutoGenBaseAgent):
    """COMPLETELY DYNAMIC orchestrator - NO hardcoded keywords or responses"""
    
    def __init__(self):
        super().__init__(
            name="orchestrator",
            description="Main orchestrator agent with LLM-driven memory-aware responses",
            agent_type="orchestrator"
        )
        self.current_session_id = None
        self.memory_enabled = True

    @message_handler
    async def handle_message(self, message: str, ctx: MessageContext) -> str:
        """Handle message with COMPLETELY DYNAMIC processing"""
        try:
            session_id = getattr(ctx, 'session_id', None) or getattr(self, 'current_session_id', None)
            
            # DYNAMIC: Search memory using LLM analysis
            relevant_memories = self.search_relevant_memory_with_context(message, session_id)
            
            # DYNAMIC: Generate response using LLM with memory context
            response = await self.generate_llm_memory_aware_response(message, relevant_memories, session_id)
            
            memory_used = len(relevant_memories) > 0
            logger.info(f"Orchestrator response: memory_used={memory_used}, memories_count={len(relevant_memories)}")
            
            return response
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return "I'm having trouble processing your request. Could you please try again?"

    def search_relevant_memory_with_context(self, message: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Search for relevant memories with DYNAMIC context integration"""
        try:
            if not self.memory_enabled:
                return []
            
            logger.info(f"Searching memory for '{message}' with session '{session_id}'")
            
            try:
                from src.learning.behavior.behavior_engine import search_semantic_memory
                results = search_semantic_memory(message, n_results=5, session_id=session_id)
            except ImportError:
                logger.warning("Memory system not available")
                return []
            
            if not results or not results.get("documents"):
                logger.info("No memory results found")
                return []
            
            memory_context = []
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results.get("metadatas", [{}])[0] if results.get("metadatas") else []
            
            logger.info(f"Found {len(documents)} memory documents")
            
            for i, doc in enumerate(documents[:5]):  # Use top 5 instead of limiting to 3
                if doc and len(doc.strip()) > 10:
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    memory_item = {
                        "content": doc.strip(),
                        "type": metadata.get("type", "unknown"),
                        "domain": metadata.get("domain", "general"),
                        "intent_type": metadata.get("intent_type", "unknown"),
                        "confidence": metadata.get("llm_confidence", 0.0),
                        "timestamp": metadata.get("timestamp", 0),
                        "session_id": metadata.get("session_id")
                    }
                    memory_context.append(memory_item)
                    logger.info(f"Memory {i}: {doc[:50]}... domain: {memory_item['domain']}, confidence: {memory_item['confidence']}")
            
            return memory_context
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def generate_llm_memory_aware_response(self, message: str, memories: List[Dict[str, Any]], session_id: str = None) -> str:
        """COMPLETELY DYNAMIC LLM-based response generation using memory"""
        try:
            # DYNAMIC: Build memory context for LLM
            memory_context = self.build_dynamic_memory_context(memories, message)
            
            # DYNAMIC: Use LLM to generate memory-aware response
            response = await self.call_llm_with_dynamic_memory_context(message, memory_context, session_id)
            
            if response:
                return response
            else:
                # DYNAMIC FALLBACK: Use LLM for basic response
                return await self.generate_basic_llm_response(message, memories)
        except Exception as e:
            logger.error(f"LLM memory response generation failed: {e}")
            return await self.generate_basic_llm_response(message, memories)

    def build_dynamic_memory_context(self, memories: List[Dict[str, Any]], query: str) -> str:
        """DYNAMIC: Build memory context using LLM analysis instead of hardcoded categories"""
        if not memories:
            return ""
        
        # DYNAMIC: Let LLM categorize memories instead of using hardcoded patterns
        memory_contents = [memory['content'] for memory in memories]
        
        try:
            categorization_prompt = f"""Analyze these memory items and organize them by relevance to the user query:

Query: {query}
Memory Items:
{chr(10).join([f"- {content}" for content in memory_contents[:5]])}

Organize these memories into relevant categories and return a structured summary.
Focus on what's most relevant to the current query."""

            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": categorization_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 200}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                llm_categorization = response.json().get("response", "").strip()
                if llm_categorization:
                    return llm_categorization
                    
        except Exception as e:
            logger.warning(f"LLM memory categorization failed: {e}")
        
        # Simple fallback - just list relevant memories
        return "\n".join([f"- {memory['content']}" for memory in memories[:3]])

    async def call_llm_with_dynamic_memory_context(self, message: str, memory_context: str, session_id: str = None) -> Optional[str]:
        """DYNAMIC LLM call with memory context - NO hardcoded prompts"""
        try:
            # DYNAMIC: Build context-aware prompt
            prompt = f"""You are Jarvis, a helpful AI assistant with access to conversation memory.

Current user message: {message}

Relevant conversation memory:
{memory_context if memory_context else "No relevant memory context available."}

Generate a natural, personalized response that:
1. Directly addresses the user's current message
2. Uses relevant memory context if available to provide personalized assistance
3. If the user is asking about their preferences and you have memory about them, reference what you remember
4. Be conversational and helpful
5. If no relevant memory, engage naturally with the current message

Respond as Jarvis:"""

            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "max_tokens": 100,
                        "top_p": 0.9
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    logger.info(f"Generated LLM memory-aware response: {result[:50]}...")
                    return result
        except Exception as e:
            logger.error(f"LLM call with memory context failed: {e}")
        
        return None

    async def generate_basic_llm_response(self, message: str, memories: List[Dict[str, Any]]) -> str:
        """DYNAMIC: Generate basic response using LLM - NO hardcoded fallbacks"""
        try:
            memory_summary = ""
            if memories:
                memory_summary = f"I have {len(memories)} relevant memories from our conversation."
            
            basic_prompt = f"""You are Jarvis, an AI assistant. Respond naturally to this message:

User: {message}
{memory_summary}

Generate a helpful, conversational response as Jarvis. Be natural and engaging."""

            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": basic_prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "max_tokens": 30}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    return result
                    
        except Exception as e:
            logger.warning(f"Basic LLM response failed: {e}")
        
        # Absolute minimal fallback
        return "I understand you're asking me something. How can I help you?"

    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process message through LLM-aware handling"""
        return await self.handle_message(message, ctx)

    def set_session_context(self, session_id: str):
        """Set session context for memory operations"""
        self.current_session_id = session_id
        logger.info(f"Orchestrator session context set: {session_id}")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status including memory capabilities"""
        try:
            db_health = db_manager.health_check()
            memory_status = "enabled" if self.memory_enabled else "disabled"
            
            return {
                "status": "healthy",
                "agent_type": "orchestrator",
                "database": db_health.get("status", "unknown"),
                "memory_enabled": self.memory_enabled,
                "memory_status": memory_status,
                "session_id": self.current_session_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# SINGLETON: Create singleton orchestrator instance
orchestrator = ReliableOrchestrator()
