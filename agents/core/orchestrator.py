"""
Reliable orchestrator agent with proper memory integration
"""
import requests
import logging
import time
from typing import Dict, Any, List, Optional
from autogen_core import RoutedAgent, message_handler, MessageContext
from agents.core.base_agent import AutoGenBaseAgent
from agents.core.database import db_manager
from config.settings import settings

logger = logging.getLogger(__name__)

class ReliableOrchestrator(AutoGenBaseAgent):
    """Enhanced orchestrator with proper memory integration following LangGraph best practices"""
    
    def __init__(self):
        super().__init__(
            name="orchestrator",
            description="Main orchestrator agent with memory-aware responses",
            agent_type="orchestrator"
        )
        self._current_session_id = None
        self._memory_enabled = True

    @message_handler
    async def handle_message(self, message: str, ctx: MessageContext) -> str:
        """Handle message with memory-aware processing"""
        try:
            # ✅ CRITICAL: Extract session from context if available
            session_id = getattr(ctx, 'session_id', None) or getattr(self, '_current_session_id', None)
            
            # ✅ OFFICIAL PATTERN: Search memory FIRST (from search results [4], [6])
            relevant_memories = self._search_relevant_memory_with_context(message, session_id)
            
            # ✅ BEST PRACTICE: Generate memory-aware response (from search result [6])
            response = await self._generate_memory_aware_response(message, relevant_memories, session_id)
            
            # ✅ LOG: Memory usage for debugging
            memory_used = len(relevant_memories) > 0
            logger.info(f"Orchestrator response: memory_used={memory_used}, memories_count={len(relevant_memories)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return "I'm having trouble processing your request. Could you please try again?"

    def _search_relevant_memory_with_context(self, message: str, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Search for relevant memories with proper context integration
        Following LangGraph semantic search patterns (search result [5])
        """
        try:
            if not self._memory_enabled:
                return []
            
            logger.info(f"Searching memory for: '{message}' with session: {session_id}")
            
            # ✅ OFFICIAL PATTERN: Use semantic search (from search result [5])
            from learning.behavior.behavior_engine import search_semantic_memory
            
            results = search_semantic_memory(message, n_results=5, session_id=session_id)
            
            if not results or not results.get('documents'):
                logger.info("No memory results found")
                return []
            
            # ✅ BEST PRACTICE: Extract and structure memory context (from search result [6])
            memory_context = []
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            
            logger.info(f"Found {len(documents)} memory documents")
            
            for i, doc in enumerate(documents[:3]):  # Limit to top 3
                if doc and len(doc.strip()) > 10:
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    memory_item = {
                        "content": doc.strip(),
                        "type": metadata.get("type", "unknown"),
                        "domain": metadata.get("domain", "general"),
                        "intent_type": metadata.get("intent_type", "unknown"),
                        "confidence": metadata.get("llm_confidence", 0.0),
                        "timestamp": metadata.get("timestamp", 0),
                        "session_id": metadata.get("session_id", "")
                    }
                    
                    memory_context.append(memory_item)
                    logger.info(f"Memory {i}: {doc[:50]}... (domain: {memory_item['domain']}, confidence: {memory_item['confidence']})")
            
            return memory_context
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def _generate_memory_aware_response(self, message: str, memories: List[Dict[str, Any]], session_id: str = None) -> str:
        """
        Generate response using memory context - Following LangGraph best practices
        Based on search results [2], [4], [6]
        """
        try:
            # ✅ OFFICIAL PATTERN: Build memory-aware context (from search result [2])
            memory_context = self._build_memory_context(memories, message)
            
            # ✅ BEST PRACTICE: Use memory-enhanced prompt (from search result [6])
            prompt = self._build_memory_enhanced_prompt(message, memory_context)
            
            # ✅ GENERATE: Response with memory integration
            response = await self._call_llm_with_memory_context(prompt, session_id)
            
            if response:
                return response
            else:
                # ✅ FALLBACK: Use memory-aware fallback
                return self._get_memory_aware_fallback(message, memories)
            
        except Exception as e:
            logger.error(f"Memory-aware response generation failed: {e}")
            return self._get_memory_aware_fallback(message, memories)

    def _build_memory_context(self, memories: List[Dict[str, Any]], query: str) -> str:
        """
        Build structured memory context following LangGraph patterns
        Based on search result [6] - semantic memory implementation
        """
        if not memories:
            return ""
        
        # ✅ BEST PRACTICE: Categorize memories by type and relevance
        preference_memories = []
        conversation_memories = []
        other_memories = []
        
        for memory in memories:
            content = memory["content"]
            memory_type = memory.get("type", "unknown")
            domain = memory.get("domain", "general")
            
            # ✅ SEMANTIC CATEGORIZATION: Based on content analysis
            if any(indicator in content.lower() for indicator in ["like", "prefer", "enjoy", "love", "favorite"]):
                preference_memories.append(f"- {content}")
            elif memory_type == "conversation":
                conversation_memories.append(f"- {content}")
            else:
                other_memories.append(f"- {content}")
        
        # ✅ STRUCTURE: Build hierarchical context
        context_parts = []
        
        if preference_memories:
            context_parts.append("**User Preferences:**")
            context_parts.extend(preference_memories[:3])  # Top 3 preferences
        
        if conversation_memories:
            context_parts.append("**Recent Conversation:**")
            context_parts.extend(conversation_memories[:2])  # Top 2 recent
        
        if other_memories:
            context_parts.append("**Additional Context:**")
            context_parts.extend(other_memories[:1])  # Top 1 other
        
        return "\n".join(context_parts) if context_parts else ""

    def _build_memory_enhanced_prompt(self, message: str, memory_context: str) -> str:
        """
        Build LLM prompt with memory integration
        Following LangGraph memory prompt patterns (search result [2])
        """
        base_prompt = f"""You are Jarvis, a helpful AI assistant with access to conversation memory.

Current user message: "{message}"

{memory_context if memory_context else "No relevant memory context available."}

Instructions:
- Use the memory context to provide personalized responses
- If the user asks about their preferences and you have relevant memory, reference it specifically
- If asking about food preferences and you know what they like, tell them what you remember
- Be conversational and natural - don't just repeat the memory verbatim
- If no relevant memory, ask clarifying questions to learn more

Respond naturally as Jarvis:"""
        
        return base_prompt

    async def _call_llm_with_memory_context(self, prompt: str, session_id: str = None) -> Optional[str]:
        """
        Call LLM with memory-enhanced prompt
        """
        try:
            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 250,
                        "top_p": 0.9
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    logger.info(f"Generated memory-aware response: {result[:50]}...")
                    return result
            
        except Exception as e:
            logger.error(f"LLM call with memory context failed: {e}")
        
        return None

    def _get_memory_aware_fallback(self, message: str, memories: List[Dict[str, Any]]) -> str:
        """
        Generate fallback response using memory context
        """
        message_lower = message.lower()
        
        # ✅ MEMORY-AWARE FALLBACKS: Use actual memory content
        if any(word in message_lower for word in ["what", "like", "eat", "food", "prefer"]):
            # Look for food preferences in memory
            food_preferences = []
            for memory in memories:
                content = memory["content"].lower()
                if any(food in content for food in ["pizza", "burger", "food", "eat", "like", "prefer"]):
                    food_preferences.append(memory["content"])
            
            if food_preferences:
                # Use the most relevant food preference
                pref = food_preferences[0]
                if "pizza" in pref.lower() and "burger" in pref.lower():
                    return "Based on our conversation, I remember you like pizza and burgers! Would you like recommendations for places that serve them?"
                elif "pizza" in pref.lower():
                    return "I remember you mentioned liking pizza! What kind of pizza do you prefer?"
                elif "burger" in pref.lower():
                    return "I recall you like burgers! Are you looking for burger recommendations?"
                else:
                    return f"Based on our previous conversation: {pref}"
            else:
                return "I'd like to learn about your food preferences! What do you like to eat?"
        
        # ✅ STANDARD FALLBACKS
        elif any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Jarvis, your AI assistant. How can I help you?"
        elif any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you with?"
        else:
            # ✅ USE MEMORY: If we have relevant context
            if memories:
                recent_memory = memories[0]["content"]
                return f"I understand you're asking me something. I remember we were discussing: {recent_memory[:50]}... How can I help you with this?"
            else:
                return "I understand you're asking me something. Could you provide more details?"

    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process message through memory-aware handling"""
        return await self.handle_message(message, ctx)

    def set_session_context(self, session_id: str):
        """Set session context for memory operations"""
        self._current_session_id = session_id
        logger.info(f"Orchestrator session context set: {session_id}")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status including memory capabilities"""
        try:
            db_health = db_manager.health_check()
            
            # ✅ CHECK: Memory system status
            memory_status = "enabled" if self._memory_enabled else "disabled"
            
            return {
                "status": "healthy",
                "agent_type": "orchestrator",
                "database": db_health.get("status", "unknown"),
                "memory_enabled": self._memory_enabled,
                "memory_status": memory_status,
                "session_id": self._current_session_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": time.time()
            }

# ✅ SINGLETON: Create singleton orchestrator instance
orchestrator = ReliableOrchestrator()
