"""
Enhanced LangGraph workflow using OFFICIAL memory patterns from LangGraph docs
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agents.core.database import db_manager
from agents.core.logging_config import log_structured
from autogen_core import MessageContext, TopicId, CancellationToken
import time
import uuid
import logging
from operator import add

logger = logging.getLogger(__name__)

# ✅ OFFICIAL PATTERN: LangGraph state schema with proper memory context
class HybridAgentState(TypedDict):
    """LangGraph state schema following official memory patterns"""
    messages: Annotated[List[BaseMessage], add]
    session_id: str
    agent_id: str
    user_input: str
    current_step: str
    reasoning: Optional[str]
    observations: Annotated[List[Dict[str, Any]], add]
    final_response: Optional[str]
    completed: bool
    context: Dict[str, Any]
    # ✅ OFFICIAL: Memory context fields from LangGraph docs
    memory_context: Dict[str, Any]
    thread_id: str

class SimpleMessageContext(MessageContext):
    """Enhanced MessageContext with session information"""
    
    def __init__(self, sender: str = "hybrid_workflow", session_id: str = None):
        super().__init__(
            sender=sender,
            topic_id=TopicId(type="default", source="workflow"),
            is_rpc=False,
            cancellation_token=CancellationToken(),
            message_id=str(uuid.uuid4())
        )
        # ✅ CRITICAL: Pass session context to AutoGen agents
        self.session_id = session_id

class HybridAgenticWorkflow:
    """Enhanced workflow using OFFICIAL LangGraph memory patterns"""
    
    def __init__(self, database=None, autogen_agents=None):
        self.database = database or db_manager
        self.autogen_agents = autogen_agents or {}
        
        # ✅ OFFICIAL PATTERN: Use MemorySaver for LangGraph persistence
        self.checkpointer = MemorySaver()
        self.workflow = self._build_hybrid_workflow()
        
        log_structured("hybrid_workflow_initialized", 
                      agent_count=len(self.autogen_agents))
    
    def _build_hybrid_workflow(self):
        """Build workflow using official LangGraph memory patterns"""
        
        def should_continue(state: HybridAgentState) -> str:
            """Determine next step in the workflow"""
            if state.get("completed", False):
                return "end"
            
            current_step = state.get("current_step", "analyze")
            
            if current_step == "analyze":
                return "plan"
            elif current_step == "plan":
                return "execute"
            elif current_step == "execute":
                return "finalize"
            else:
                return "end"
        
        # ✅ OFFICIAL: StateGraph with proper memory state
        workflow = StateGraph(HybridAgentState)
        
        # ✅ OFFICIAL PATTERN: Memory-aware nodes from LangGraph docs
        workflow.add_node("analyze_with_memory", self._analyze_with_session_memory)
        workflow.add_node("plan_with_agents", self._plan_with_available_agents)
        workflow.add_node("execute_with_autogen", self._execute_via_autogen_agents)
        workflow.add_node("finalize_response", self._finalize_and_store)
        
        # Define the flow
        workflow.add_edge(START, "analyze_with_memory")
        workflow.add_conditional_edges(
            "analyze_with_memory",
            should_continue,
            {
                "plan": "plan_with_agents",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "plan_with_agents", 
            should_continue,
            {
                "execute": "execute_with_autogen",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "execute_with_autogen",
            should_continue,
            {
                "finalize": "finalize_response", 
                "end": END
            }
        )
        workflow.add_edge("finalize_response", END)
        
        # ✅ OFFICIAL: Compile with checkpointer for memory persistence
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _analyze_with_session_memory(self, state: HybridAgentState) -> Dict[str, Any]:
        """Memory analysis using OFFICIAL LangGraph session patterns"""
        log_structured("hybrid_analysis_start", 
                      session_id=state["session_id"], 
                      agent_id=state["agent_id"])
        
        try:
            # ✅ OFFICIAL PATTERN: Use session context for memory search
            session_id = state.get("session_id") or state.get("thread_id")
            
            # Search memory with proper session context
            from learning.behavior.behavior_engine import search_semantic_memory
            memory_results = search_semantic_memory(
                state["user_input"], 
                n_results=5, 
                session_id=session_id
            )
            
            # ✅ OFFICIAL: Store memory context in state following LangGraph patterns
            memory_context = {
                "retrieved_memories": memory_results,
                "session_id": session_id,
                "search_query": state["user_input"],
                "memory_count": len(memory_results.get('documents', [[]])[0]) if memory_results else 0
            }
            
            reasoning = f"""
            HYBRID ANALYSIS WITH SESSION MEMORY:
            User Input: "{state['user_input']}"
            Session ID: {session_id}
            Available AutoGen Agents: {list(self.autogen_agents.keys())}
            Memory Results: {memory_context['memory_count']} relevant memories found
            """
            
            # ✅ RETURN: State updates with memory context
            return {
                "current_step": "analyze",
                "reasoning": reasoning,
                "memory_context": memory_context,
                "context": {"memory_results": memory_results}
            }
            
        except Exception as e:
            log_structured("analysis_error", error=str(e), session_id=state["session_id"])
            return {
                "current_step": "analyze", 
                "reasoning": f"Basic analysis of: {state['user_input']}",
                "memory_context": {},
                "context": {}
            }
    
    async def _plan_with_available_agents(self, state: HybridAgentState) -> Dict[str, Any]:
        """Planning with memory-aware agent selection"""
        log_structured("hybrid_planning_start", session_id=state["session_id"])
        
        try:
            # Determine which agent to use based on input
            user_input = state["user_input"].lower()
            
            if any(keyword in user_input for keyword in ["weather", "temperature", "forecast"]):
                selected_agent = "weather_agent"
            elif any(keyword in user_input for keyword in ["routine", "schedule", "plan"]):
                selected_agent = "routine_agent" 
            else:
                selected_agent = "orchestrator"
            
            log_structured("hybrid_planning_completed", 
                         selected_agent=selected_agent,
                         session_id=state["session_id"])
            
            # ✅ RETURN: State updates with session context for agents
            return {
                "current_step": "plan",
                "context": {
                    **state.get("context", {}),
                    "selected_agent": selected_agent,
                    "plan_created": True,
                    # ✅ CRITICAL: Pass session context to agents
                    "session_id": state.get("session_id"),
                    "thread_id": state.get("thread_id")
                }
            }
            
        except Exception as e:
            log_structured("planning_error", error=str(e), session_id=state["session_id"])
            return {
                "current_step": "plan",
                "context": {
                    **state.get("context", {}), 
                    "selected_agent": "orchestrator",
                    "session_id": state.get("session_id")
                }
            }
    
    async def _execute_via_autogen_agents(self, state: HybridAgentState) -> Dict[str, Any]:
        """Execute using AutoGen agents with PROPER session context"""
        log_structured("autogen_execution_start", session_id=state["session_id"])
        
        try:
            # Get selected agent and session context
            context_data = state.get("context", {})
            selected_agent = context_data.get("selected_agent", "orchestrator")
            session_id = context_data.get("session_id") or state.get("session_id")
            
            # ✅ CRITICAL FIX: Store memory BEFORE generating response
            await self._store_current_interaction_before_response(state)
            
            # ✅ OFFICIAL PATTERN: Call AutoGen agent with session context
            result = await self._call_autogen_agent_with_session_context(
                state["user_input"], 
                selected_agent,
                session_id,
                state.get("memory_context", {})
            )
            
            # Create observation record
            observation = {
                "agent_type": selected_agent,
                "result": result,
                "success": True,
                "timestamp": time.time(),
                "session_id": session_id
            }
            
            log_structured("autogen_action_completed",
                         action_type=selected_agent,
                         success=True,
                         result_length=len(str(result)),
                         session_id=session_id)
            
            # ✅ RETURN: State updates with session-aware observation
            return {
                "current_step": "execute",
                "observations": [observation],
                "context": {
                    **context_data,
                    "execution_result": result,
                    "execution_success": True
                }
            }
            
        except Exception as e:
            # Record failed observation
            failed_observation = {
                "agent_type": state.get("context", {}).get("selected_agent", "unknown"),
                "result": None,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
                "session_id": state.get("session_id")
            }
            
            log_structured("autogen_action_failed",
                         action_type=state.get("context", {}).get("selected_agent", "unknown"),
                         error=str(e),
                         session_id=state["session_id"])
            
            return {
                "current_step": "execute",
                "observations": [failed_observation],
                "context": {
                    **state.get("context", {}),
                    "execution_success": False,
                    "execution_error": str(e)
                }
            }
    
    async def _store_current_interaction_before_response(self, state: HybridAgentState):
        """Store current interaction BEFORE generating response (CRITICAL FIX)"""
        try:
            # ✅ CRITICAL: Store the current user input immediately so it's available for memory search
            session_id = state.get("session_id")
            user_input = state.get("user_input", "")
            
            if user_input and session_id:
                from learning.behavior.behavior_engine import add_to_semantic_memory
                
                # Store user input with session context
                interaction_content = f"User: {user_input}"
                metadata = {
                    "type": "user_message",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "agent_id": state.get("agent_id", "hybrid_main")
                }
                
                add_to_semantic_memory(interaction_content, metadata)
                logger.info(f"Stored user input before response generation: {user_input[:50]}")
                
        except Exception as e:
            logger.warning(f"Failed to store interaction before response: {e}")
    
    async def _call_autogen_agent_with_session_context(self, user_input: str, agent_type: str, session_id: str, memory_context: Dict[str, Any]) -> str:
        """Call AutoGen agent with PROPER session context"""
        
        try:
            # ✅ CRITICAL: Create context with session information
            context = SimpleMessageContext("hybrid_workflow", session_id)
            
            if agent_type == "weather_agent" and "weather" in self.autogen_agents:
                agent = self.autogen_agents["weather"]
                # ✅ OFFICIAL: Set session context on agent
                if hasattr(agent, '_current_session_id'):
                    agent._current_session_id = session_id
                result = await agent.process_message(user_input, context)
                logger.info(f"Weather agent response: {result[:100]}")
                return result
            
            elif agent_type == "routine_agent" and "routine" in self.autogen_agents:
                agent = self.autogen_agents["routine"]
                if hasattr(agent, '_current_session_id'):
                    agent._current_session_id = session_id
                result = await agent.process_message(user_input, context)
                logger.info(f"Routine agent response: {result[:100]}")
                return result
            
            else:
                # ✅ CRITICAL: Use orchestrator with session context
                if "orchestrator" in self.autogen_agents:
                    agent = self.autogen_agents["orchestrator"]
                    # ✅ OFFICIAL PATTERN: Set session context
                    agent._current_session_id = session_id
                    result = await agent.process_message(user_input, context)
                else:
                    from agents.core.orchestrator import orchestrator
                    orchestrator._current_session_id = session_id
                    result = await orchestrator.process_message(user_input, context)
                
                logger.info(f"Orchestrator response: {result[:100]}")
                return result
                
        except Exception as e:
            raise Exception(f"Agent execution failed for {agent_type}: {str(e)}")
    
    async def _finalize_and_store(self, state: HybridAgentState) -> Dict[str, Any]:
        """Finalize response and store complete interaction with session context"""
        
        try:
            # Extract the best response from observations
            observations = state.get("observations", [])
            successful_obs = [obs for obs in observations if obs.get("success", False)]
            
            final_response = ""
            
            if successful_obs:
                # Use the result from the most recent successful observation
                latest_success = successful_obs[-1]
                result = latest_success.get("result", "")
                
                if result and str(result).strip():
                    result_str = str(result).strip()
                    # Skip memory storage confirmations
                    if not result_str.startswith("Interaction stored"):
                        final_response = result_str
            
            # Fallback to classified response if no good result
            if not final_response:
                input_type = self._classify_input_type(state["user_input"])
                fallback_responses = {
                    "greeting": "Hello! I'm Jarvis, your AI assistant. How can I help you?",
                    "weather_query": "I'd be happy to help with weather information. Could you specify a location?",
                    "routine_query": "I can help you with routines and scheduling. What would you like to plan?",
                    "general_query": "I understand you're asking me something. How can I help you?"
                }
                final_response = fallback_responses.get(input_type, "How can I help you?")
            
            # ✅ OFFICIAL PATTERN: Store complete interaction with session context
            session_id = state.get("session_id")
            if session_id:
                try:
                    from learning.behavior.behavior_engine import add_to_semantic_memory
                    interaction_content = f"User: {state['user_input']}\nJarvis: {final_response}"
                    metadata = {
                        "type": "conversation",
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "agent_id": state.get("agent_id", "hybrid_main")
                    }
                    add_to_semantic_memory(interaction_content, metadata)
                except Exception as e:
                    logger.warning(f"Memory storage failed: {e}")
            
            log_structured("hybrid_workflow_completed",
                         response_length=len(final_response),
                         total_observations=len(observations),
                         session_id=session_id,
                         completed=True)
            
            # ✅ RETURN: Final state updates
            return {
                "current_step": "finalize",
                "final_response": final_response,
                "completed": True,
                "messages": [AIMessage(content=final_response)]
            }
            
        except Exception as e:
            log_structured("finalization_error", error=str(e), session_id=state.get("session_id"))
            
            error_response = "I encountered an error while finalizing the response."
            return {
                "current_step": "finalize",
                "final_response": error_response,
                "completed": True,
                "messages": [AIMessage(content=error_response)]
            }
    
    def _classify_input_type(self, user_input: str) -> str:
        """Classify user input type"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["weather", "temperature", "forecast"]):
            return "weather_query"
        elif any(word in user_lower for word in ["routine", "schedule", "plan"]):
            return "routine_query"
        elif any(word in user_lower for word in ["hello", "hi", "hey"]):
            return "greeting"
        else:
            return "general_query"
    
    async def run(self, state_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run workflow using OFFICIAL LangGraph memory patterns"""
        try:
            # ✅ OFFICIAL PATTERN: Create proper initial state with thread context
            initial_state: HybridAgentState = {
                "messages": [HumanMessage(content=state_input["user_input"])],
                "session_id": state_input["session_id"],
                "agent_id": state_input["agent_id"],
                "user_input": state_input["user_input"],
                "current_step": "start",
                "reasoning": None,
                "observations": [],
                "final_response": None,
                "completed": False,
                "context": {},
                "memory_context": {},
                "thread_id": state_input["session_id"]  # ✅ OFFICIAL: Use session_id as thread_id
            }
            
            logger.info(f"Workflow start: session_id={initial_state['session_id']}")
            
            # ✅ OFFICIAL PATTERN: Use config with thread_id for memory persistence
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
            
            # Run workflow with proper memory context
            result = await self.workflow.ainvoke(initial_state, config)
            
            logger.info(f"Workflow completed: {result.get('completed', False)}")
            logger.info(f"Final response: {result.get('final_response', '')[:100]}")
            logger.info(f"Observations: {len(result.get('observations', []))}")
            
            return result
            
        except Exception as e:
            log_structured("hybrid_workflow_error", error=str(e), 
                         session_id=state_input.get("session_id", "unknown"))
            
            return {
                **state_input,
                "final_response": f"Workflow error: {str(e)}",
                "completed": True,
                "observations": [],
                "messages": [],
                "current_step": "error",
                "reasoning": None,
                "context": {},
                "memory_context": {},
                "thread_id": state_input.get("session_id", "unknown")
            }
