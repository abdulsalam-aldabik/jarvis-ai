import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional
from agents.core.base_agent import BaseAgent, AgentCapability, AgentTask, AgentResponse, agent_registry, IntelligentLLMInterface
from agents.core.database import db_manager
from learning.behavior.behavior_engine import add_to_semantic_memory, search_semantic_memory
from config.settings import settings

class IntelligentOrchestrator(BaseAgent):
    """Intelligent orchestrator that uses LLM reasoning for coordination"""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="coordinate_intelligent_response",
                description="Coordinate intelligent responses using LLM and specialized agents",
                input_schema={"user_request": "string", "context": "object"},
                output_schema={"intelligent_response": "string", "reasoning": "string"}
            ),
            AgentCapability(
                name="learn_from_interaction",
                description="Learn from user interactions and improve responses",
                input_schema={"interaction": "object"},
                output_schema={"learned": "boolean", "insights": "array"}
            )
        ]
        
        super().__init__(
            agent_id="orchestrator_agent",
            agent_type="orchestrator",
            capabilities=capabilities
        )
        self.agent_registry = agent_registry
    
    
    async def execute_intelligent_plan(self, task: AgentTask, reasoning: str) -> Any:
        """Execute intelligent coordination plan"""
        user_request = task.content
        context = task.context
        
        try:
            # Step 1: Analyze the request intelligently
            analysis = await self._analyze_request_intelligently(user_request, context)
            
            # Step 2: Check if we need specialized agents
            if analysis.get("needs_specialized_agent"):
                specialized_result = await self._coordinate_with_specialists(user_request, analysis, context)
                if specialized_result and specialized_result.get("success"):
                    return specialized_result
            
            # Step 3: Use semantic memory for context
            memory_context = await self._get_relevant_memory(user_request)
            
            # Step 4: Generate intelligent response using LLM
            intelligent_response = await self._generate_intelligent_response(
                user_request, analysis, memory_context, context
            )
            
            # Step 5: Learn from this interaction
            await self._learn_from_interaction(user_request, intelligent_response, context)
            
            return {
                "workflow_result": intelligent_response,
                "analysis": analysis,
                "memory_used": bool(memory_context),
                "reasoning": reasoning
            }
            
        except Exception as e:
            # Fallback to basic LLM response
            fallback_response = await self.llm.think(
                f"The user asked: {user_request}. Please provide a helpful response.",
                context, self.agent_id
            )
            
            return {
                "workflow_result": fallback_response,
                "analysis": {"error": str(e)},
                "fallback_used": True
            }
    
    async def _analyze_request_intelligently(self, user_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze what the user really needs"""
        analysis_prompt = f"""
        Analyze this user request intelligently:
        
        Request: "{user_request}"
        Context: {json.dumps(context, indent=2)}
        
        Available specialized agents: {list(agent_registry.list_agents())}
        
        Determine:
        1. What does the user really want?
        2. What type of response would be most helpful?
        3. Should I use a specialized agent? Which one?
        4. What's the intent and sentiment?
        5. Is this a question, command, statement, or conversation?
        
        Respond in JSON format:
        {{
            "intent": "string",
            "request_type": "question|command|statement|conversation",
            "sentiment": "positive|neutral|negative",
            "needs_specialized_agent": true/false,
            "recommended_agent": "agent_id or null",
            "complexity": "simple|medium|complex",
            "expected_response_type": "informational|actionable|conversational"
        }}
        """
        
        analysis_text = await self.llm.think(analysis_prompt, context, self.agent_id)
        
        try:
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback analysis
                analysis = {
                    "intent": "general",
                    "request_type": "question",
                    "sentiment": "neutral",
                    "needs_specialized_agent": False,
                    "complexity": "medium",
                    "expected_response_type": "conversational"
                }
        except:
            analysis = {
                "intent": "general", 
                "needs_specialized_agent": False,
                "analysis_text": analysis_text
            }
        
        return analysis
    
    async def _coordinate_with_specialists(self, user_request: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently coordinate with specialized agents"""
        recommended_agent = analysis.get("recommended_agent")
        
        # Determine which agent to use
        if not recommended_agent:
            if "weather" in user_request.lower():
                recommended_agent = "weather_agent"
            elif any(word in user_request.lower() for word in ["routine", "schedule", "plan", "habit"]):
                recommended_agent = "routine_agent"
        
        # FIX: Use the AgentRegistry properly
        if recommended_agent and recommended_agent in self.agent_registry.list_agents():
            try:
                agent_instance = self.agent_registry.get_agent(recommended_agent)
                if agent_instance:
                    specialist_task = AgentTask(
                        task_id=str(uuid.uuid4()),
                        task_type="intelligent_request",
                        content=user_request,
                        context=context,
                        requester_id=self.agent_id
                    )
                    
                    response = await agent_instance.handle_task_direct(specialist_task)
                    
                    if response.success:
                        return {
                            "success": True,
                            "workflow_result": response.result,
                            "specialist_used": recommended_agent,
                            "specialist_reasoning": response.reasoning
                        }
            except Exception as e:
                db_manager.log_event("ERROR", f"Specialist coordination failed: {str(e)}", 
                                    {"agent": recommended_agent}, self.agent_id)
        
        return {"success": False}

    
    async def _get_relevant_memory(self, user_request: str) -> List[Dict[str, Any]]:
        """Get relevant memory using dynamic search strategy"""
        try:
            # Let the LLM determine the best search strategy
            search_strategy = await self._determine_search_strategy(user_request)
            
            # Execute the search based on LLM strategy
            memories = await self._execute_dynamic_search(user_request, search_strategy)
            
            return memories
            
        except Exception as e:
            db_manager.log_event("WARNING", f"Dynamic memory search failed: {str(e)}", {}, self.agent_id)
            return []

    async def _determine_search_strategy(self, user_request: str) -> str:
        """Let LLM determine how to search memory for this request"""
        
        strategy_prompt = f"""
        User Request: "{user_request}"
        
        Determine the best strategy for searching previous interactions to find relevant context:
        
        1. What keywords or concepts should be searched for?
        2. What types of previous interactions would be most relevant?
        3. Should the search focus on recent interactions or specific topics?
        4. What search terms would be most effective?
        
        Provide a search strategy with specific terms and approach.
        """
        
        return await self.llm.think(strategy_prompt, {"task": "search_strategy"}, self.agent_id)

    async def _execute_dynamic_search(self, user_request: str, search_strategy: str) -> List[Dict[str, Any]]:
        """Execute memory search based on LLM-determined strategy"""
        
        # Extract search terms from the strategy using LLM
        terms_prompt = f"""
        Based on this search strategy:
        {search_strategy}
        
        Extract 3-5 specific search terms that should be used to find relevant memories.
        Return them as a simple list, one term per line.
        """
        
        terms_response = await self.llm.think(terms_prompt, {"task": "term_extraction"}, self.agent_id)
        
        # Parse search terms
        search_terms = [term.strip() for term in terms_response.split('\n') if term.strip()]
        if not search_terms:
            search_terms = [user_request]  # Fallback to original request
        
        # Search with multiple terms
        all_memories = []
        for term in search_terms[:3]:  # Limit to top 3 terms
            results = search_semantic_memory(term, n_results=3)
            
            if results and results.get('documents'):
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                
                for i, doc in enumerate(documents):
                    if i < len(metadatas):
                        all_memories.append({
                            "content": doc,
                            "metadata": metadatas[i],
                            "search_term": term,
                            "relevance": "high" if i == 0 else "medium"
                        })
        
        # Remove duplicates and return top results
        seen_content = set()
        unique_memories = []
        for memory in all_memories:
            if memory["content"] not in seen_content:
                seen_content.add(memory["content"])
                unique_memories.append(memory)
                if len(unique_memories) >= 5:
                    break
        
        return unique_memories


    
    
    async def _generate_intelligent_response(self, user_request: str, analysis: Dict[str, Any], memory_context: List[Dict], context: Dict[str, Any]) -> str:
        """Generate truly intelligent response without any hardcoding"""
        
        # Step 1: Let the LLM analyze what information is relevant from memory
        memory_analysis = await self._analyze_memory_relevance(user_request, memory_context)
        
        # Step 2: Let the LLM determine the appropriate response style
        response_style = await self._determine_response_style(user_request, analysis, memory_analysis)
        
        # Step 3: Generate the response using dynamic prompting
        response = await self._generate_dynamic_response(user_request, analysis, memory_analysis, response_style, context)
        
        return response

    async def _analyze_memory_relevance(self, user_request: str, memory_context: List[Dict]) -> str:
        """Let LLM analyze which memories are relevant and why"""
        if not memory_context:
            return "No previous context available."
        
        memory_analysis_prompt = f"""
        User Request: "{user_request}"
        
        Available Memory Context:
        {json.dumps(memory_context, indent=2)}
        
        Analyze this memory context and determine:
        1. Which memories are directly relevant to answering the user's request?
        2. What specific information can be extracted that would help personalize the response?
        3. Are there any patterns or preferences that emerge from the memories?
        4. What context should be considered when responding?
        
        Provide a concise analysis focusing only on what's relevant to this specific request.
        """
        
        return await self.llm.think(memory_analysis_prompt, {"task": "memory_analysis"}, self.agent_id)

    async def _determine_response_style(self, user_request: str, analysis: Dict[str, Any], memory_analysis: str) -> str:
        """Let LLM determine the appropriate response style dynamically"""
        
        style_prompt = f"""
        User Request: "{user_request}"
        Request Analysis: {json.dumps(analysis, indent=2)}
        Memory Analysis: {memory_analysis}
        
        Based on this information, determine the most appropriate way to respond:
        - What tone should be used?
        - How detailed should the response be?
        - Should it be conversational, informational, or actionable?
        - How should the memory context be incorporated?
        - What would make this response most helpful to the user?
        
        Provide guidance for how to structure and style the response.
        """
        
        return await self.llm.think(style_prompt, {"task": "style_determination"}, self.agent_id)

    async def _generate_dynamic_response(self, user_request: str, analysis: Dict[str, Any], memory_analysis: str, response_style: str, context: Dict[str, Any]) -> str:
        """Generate the final response using completely dynamic prompting"""
        
        dynamic_prompt = f"""
        You are Jarvis, an intelligent AI assistant. You need to respond to this user request in the most helpful way possible.
        
        User Request: "{user_request}"
        
        Analysis of Request: {json.dumps(analysis, indent=2)}
        
        Relevant Memory Analysis: {memory_analysis}
        
        Response Style Guidance: {response_style}
        
        Additional Context: {json.dumps(context, indent=2)}
        
        Generate a response that:
        - Directly addresses what the user is asking
        - Uses the relevant information from your analysis
        - Follows the style guidance appropriately
        - Feels natural and intelligent
        - Is personalized based on available context
        
        Your response:
        """
        
        return await self.llm.think(dynamic_prompt, {
            "request_type": "final_response",
            "has_memory": bool(memory_analysis and "No previous context" not in memory_analysis)
        }, self.agent_id)

        
    async def _learn_from_interaction(self, user_request: str, response: str, context: Dict[str, Any]):
        """Learn from interaction dynamically without hardcoded rules"""
        try:
            # Let the LLM determine what should be learned from this interaction
            learning_analysis = await self._analyze_learning_opportunities(user_request, response, context)
            
            # Store the interaction with LLM-determined metadata
            await self._store_learned_information(user_request, response, learning_analysis, context)
            
            db_manager.log_event("INFO", "Dynamic learning completed", {
                "learning_analysis_length": len(learning_analysis)
            }, self.agent_id)
            
        except Exception as e:
            db_manager.log_event("WARNING", f"Dynamic learning failed: {str(e)}", {}, self.agent_id)

    async def _analyze_learning_opportunities(self, user_request: str, response: str, context: Dict[str, Any]) -> str:
        """Let LLM determine what should be learned from this interaction"""
        
        learning_prompt = f"""
        Analyze this interaction to determine what information should be stored for future reference:
        
        User Request: "{user_request}"
        System Response: "{response}"
        Context: {json.dumps(context, indent=2)}
        
        Determine:
        1. Does this interaction contain any user preferences, facts, or personal information?
        2. What categories of information are present (if any)?
        3. What would be useful to remember for future interactions?
        4. How should this information be categorized and tagged for future retrieval?
        5. What keywords or concepts are most important for finding this information later?
        
        Provide a structured analysis of what should be learned and how it should be categorized.
        """
        
        return await self.llm.think(learning_prompt, {"task": "learning_analysis"}, self.agent_id)

    async def _store_learned_information(self, user_request: str, response: str, learning_analysis: str, context: Dict[str, Any]):
        """Store information based on LLM analysis"""
        
        # Let the LLM determine the storage format
        storage_prompt = f"""
        Based on this learning analysis:
        {learning_analysis}
        
        Generate a JSON object with metadata for storing this interaction. Include:
        - type: the type of interaction or information
        - categories: relevant categories for future search
        - keywords: important keywords for retrieval
        - importance: how important this is to remember (1-10)
        - any other relevant metadata fields
        
        Return only valid JSON:
        """
        
        metadata_response = await self.llm.think(storage_prompt, {"task": "metadata_generation"}, self.agent_id)
        
        # Try to parse the LLM response as JSON, fallback to basic metadata
        try:
            import re
            json_match = re.search(r'\{.*\}', metadata_response, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
            else:
                metadata = {"type": "conversation", "dynamic_analysis": learning_analysis}
        except:
            metadata = {
                "type": "conversation", 
                "timestamp": time.time(),
                "dynamic_analysis": learning_analysis[:200]  # Truncate for storage
            }
        
        # Ensure timestamp is always included
        metadata["timestamp"] = time.time()
        
        # Store the interaction
        interaction_content = f"User: {user_request}\nJarvis: {response}\nLearning: {learning_analysis}"
        add_to_semantic_memory(interaction_content, metadata)

# Create intelligent orchestrator instance
orchestrator = IntelligentOrchestrator()
