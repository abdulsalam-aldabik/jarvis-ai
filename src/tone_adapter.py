from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from config.settings import settings
import requests
import json

class ToneStyle(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"

@dataclass
class ResponseContext:
    """Context for response generation"""
    user_message: str
    agent_type: str
    raw_data: Dict[str, Any]
    user_preferences: Dict[str, Any]
    communication_style: str
    conversation_history: str
    time_context: str
    intent: str

class ToneAdapter:
    def __init__(self):
        self.style_prompts = {
            ToneStyle.FORMAL: "Respond in a professional, polite, and formal tone. Use complete sentences and proper grammar.",
            ToneStyle.CASUAL: "Respond in a relaxed, conversational tone. Use contractions and informal language.",
            ToneStyle.ENTHUSIASTIC: "Respond with energy and excitement. Use exclamation points and positive language.",
            ToneStyle.PROFESSIONAL: "Respond in a business-appropriate, competent tone.",
            ToneStyle.FRIENDLY: "Respond in a warm, approachable, and helpful tone."
        }
    
    async def generate_response(self, response_context: ResponseContext, shared_context: Any) -> str:
        """Generate response using ResponseContext - MAIN METHOD"""
        base_content = response_context.raw_data.get("response", response_context.raw_data.get("draft", ""))
        
        user_context = {
            'user_preferences': response_context.user_preferences,
            'communication_style': response_context.communication_style,
            'conversation_history': response_context.conversation_history,
            'time_context': response_context.time_context,
            'intent': response_context.intent,
            'agent_type': response_context.agent_type
        }
        
        tone_style = self._map_communication_style_to_tone(response_context.communication_style)
        return await self.generate_contextual_response(base_content, user_context, tone_style)
    
    def _map_communication_style_to_tone(self, comm_style: str) -> ToneStyle:
        """Map communication style to tone style"""
        mapping = {
            'formal': ToneStyle.FORMAL,
            'casual': ToneStyle.CASUAL,
            'enthusiastic': ToneStyle.ENTHUSIASTIC,
            'professional': ToneStyle.PROFESSIONAL,
            'friendly': ToneStyle.FRIENDLY
        }
        return mapping.get(comm_style, ToneStyle.FRIENDLY)
    
    async def generate_contextual_response(self, base_content: str, user_context: Dict[str, Any], 
                                         tone_style: ToneStyle = ToneStyle.FRIENDLY) -> str:
        """Generate LLM response with adaptive tone and context awareness"""
        
        # Build context-aware prompt
        prompt = self._build_dynamic_prompt(base_content, user_context, tone_style)
        
        # Call LLM for dynamic response generation
        response = await self._call_llm(prompt)
        return response
    
    def _build_dynamic_prompt(self, content: str, context: Dict[str, Any], tone: ToneStyle) -> str:
        user_prefs = context.get('user_preferences', {})
        conversation_history = context.get('conversation_history', '')
        time_context = context.get('time_context', {})
        
        prompt = f"""
You are Jarvis, an intelligent AI assistant. Generate a response that:

TONE INSTRUCTION: {self.style_prompts[tone]}

USER CONTEXT:
- Communication Style: {context.get('communication_style', 'friendly')}
- Previous Preferences: {json.dumps(user_prefs, indent=2) if user_prefs else 'None learned yet'}
- Recent Conversation: {conversation_history}
- Current Time: {time_context.get('time_period', 'unknown')}
- Intent: {context.get('intent', 'general')}

RESPONSE CONTENT TO PERSONALIZE:
{content}

REQUIREMENTS:
- Reference user preferences when relevant
- Maintain conversation continuity
- Be proactive and helpful
- Ask follow-up questions when appropriate
- Keep response natural and conversational
- Adapt to the detected communication style

Generate a personalized response:
"""
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM for response generation"""
        try:
            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 300,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return "I apologize, but I'm having trouble generating a response right now."
                
        except Exception as e:
            return f"I encountered an issue: {str(e)}"

# Global instance
tone_adapter = ToneAdapter()
