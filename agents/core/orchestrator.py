import asyncio
import time
import json
from typing import Dict, Any, List
from autogen_core import MessageContext
from agents.core.base_agent import AutoGenBaseAgent
from agents.core.database import db_manager
from learning.behavior.behavior_engine import add_to_semantic_memory, search_semantic_memory
import requests
from config.settings import settings

class ReliableOrchestrator(AutoGenBaseAgent):
    """Reliable AutoGen orchestrator with proper memory"""
    
    def __init__(self):
        super().__init__("orchestrator", "Reliable AutoGen orchestrator")
    
    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process message reliably"""
        try:
            # Step 1: Check if this needs a specialist
            if await self._needs_weather_agent(message):
                response = await self._delegate_to_weather(message)
                if response:
                    await self._store_interaction(message, response)
                    return response
            
            if await self._needs_routine_agent(message):
                response = await self._delegate_to_routine(message)
                if response:
                    await self._store_interaction(message, response)
                    return response
            
            # Step 2: Search relevant memory
            relevant_memory = self._search_relevant_memory(message)
            
            # Step 3: Generate response with LLM
            response = await self._generate_response_with_memory(message, relevant_memory)
            
            # Step 4: Store this interaction
            await self._store_interaction(message, response)
            
            return response
            
        except Exception as e:
            db_manager.log_event("ERROR", f"Orchestrator failed: {str(e)}", {}, self.agent_id)
            return "I'm having trouble processing that request. Please try again."
    
    async def _needs_weather_agent(self, message: str) -> bool:
        """Simple, reliable weather detection"""
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "degrees"]
        return any(keyword in message.lower() for keyword in weather_keywords)
    
    async def _needs_routine_agent(self, message: str) -> bool:
        """Simple, reliable routine detection"""
        routine_keywords = ["routine", "schedule", "plan", "habit", "morning", "evening", "daily"]
        return any(keyword in message.lower() for keyword in routine_keywords)
    
    async def _delegate_to_weather(self, message: str) -> str:
        """Delegate to weather agent"""
        try:
            from agents.specialized.weather_agent import ReliableWeatherAgent
            agent = ReliableWeatherAgent()
            
            class MockContext:
                sender = "orchestrator"
            
            return await agent.process_message(message, MockContext())
        except Exception as e:
            return f"Weather service is temporarily unavailable: {str(e)[:50]}"
    
    async def _delegate_to_routine(self, message: str) -> str:
        """Delegate to routine agent"""
        try:
            from agents.specialized.routine_agent import ReliableRoutineAgent
            agent = ReliableRoutineAgent()
            
            class MockContext:
                sender = "orchestrator"
            
            return await agent.process_message(message, MockContext())
        except Exception as e:
            return "Routine planning is temporarily unavailable. Please try again later."
    
    def _search_relevant_memory(self, message: str) -> List[str]:
        """Search for relevant memories with reliability checks"""
        try:
            results = search_semantic_memory(message, n_results=3)
            
            if not results or not results.get('documents'):
                return []
            
            # Extract only relevant content
            relevant_memories = []
            documents = results['documents'][0]
            
            for doc in documents[:2]:  # Limit to 2 most relevant
                if doc and len(doc.strip()) > 10:  # Only meaningful content
                    relevant_memories.append(doc.strip())
            
            return relevant_memories
            
        except Exception as e:
            db_manager.log_event("WARNING", f"Memory search failed: {str(e)}", {}, self.agent_id)
            return []
    
    async def _generate_response_with_memory(self, message: str, memories: List[str]) -> str:
        """Generate response using LLM with memory context"""
        try:
            # Build context
            context = ""
            if memories:
                context = f"\nRelevant context from previous conversations:\n"
                for i, memory in enumerate(memories, 1):
                    context += f"{i}. {memory[:150]}...\n"
            
            # Create prompt
            prompt = f"""You are Jarvis, a helpful AI assistant. Respond concisely and accurately.

User: {message}{context}

Jarvis: """
            
            # Call LLM
            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 150,  # Keep responses concise
                        "top_p": 0.9
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    return result
            
        except Exception as e:
            db_manager.log_event("ERROR", f"LLM generation failed: {str(e)}", {}, self.agent_id)
        
        # Fallback responses
        return self._get_fallback_response(message)
    
    def _get_fallback_response(self, message: str) -> str:
        """Reliable fallback responses"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Jarvis, your AI assistant. How can I help you?"
        
        elif any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you with?"
        
        elif any(word in message_lower for word in ["what", "tell me", "about"]):
            return "I'm here to help! I can assist with weather information, routine planning, and general questions."
        
        else:
            return "I understand you're asking me something. Could you please rephrase that or ask about weather, routines, or general topics?"
    
    async def _store_interaction(self, user_message: str, response: str):
        """Store interaction in memory"""
        try:
            interaction_content = f"User: {user_message}\nJarvis: {response}"
            
            metadata = {
                "type": "conversation",
                "timestamp": time.time()
            }
            
            add_to_semantic_memory(interaction_content, metadata)
            
        except Exception as e:
            db_manager.log_event("WARNING", f"Failed to store interaction: {str(e)}", {}, self.agent_id)

# Create global orchestrator
orchestrator = ReliableOrchestrator()
