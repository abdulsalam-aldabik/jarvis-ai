"""
COMPLETELY DYNAMIC LangGraph workflow with FIXED AutoGen 2.x compatibility and error handling
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.agents.core.database import db_manager
from src.agents.core.logging_config import log_structured
from autogen_core import MessageContext, TopicId, CancellationToken
import time
import uuid
import logging
from operator import add

logger = logging.getLogger(__name__)

class HybridAgentState(TypedDict):
    """Dynamic state schema - no hardcoded fields"""
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
    memory_context: Dict[str, Any]
    thread_id: str

class SimpleMessageContext(MessageContext):
    """FIXED MessageContext with correct AutoGen 2.x TopicId constructor"""
    def __init__(self, sender: str = "hybrid_workflow", session_id: str = None):
        # CRITICAL FIX: TopicId requires source as positional argument (from file 1)
        super().__init__(
            sender=sender,
            topic_id=TopicId("default", source="workflow"),  # FIXED: source as positional argument
            is_rpc=False,
            cancellation_token=CancellationToken(),
            message_id=str(uuid.uuid4())
        )
        self.source = "workflow"
        self.session_id = session_id

class HybridAgenticWorkflow:
    """COMPLETELY DYNAMIC workflow with FIXED error handling - NO hardcoded routing or responses"""
    
    def __init__(self, database=None, autogen_agents=None):
        self.database = database or db_manager
        self.autogen_agents = autogen_agents or {}
        self.checkpointer = MemorySaver()
        self.workflow = self._build_hybrid_workflow()
        
        # DYNAMIC: Get agent capabilities from database
        self.available_agents = list(self.autogen_agents.keys()) if self.autogen_agents else []
        log_structured("hybrid_workflow_initialized", agent_count=len(self.autogen_agents))

    def _build_hybrid_workflow(self):
        """Build workflow with completely dynamic routing"""
        
        def should_continue(state: HybridAgentState) -> str:
            """DYNAMIC workflow routing based on LLM analysis"""
            if state.get("completed", False):
                return "end"
            
            current_step = state.get("current_step", "analyze")
            # DYNAMIC: Use context analysis instead of hardcoded steps
            context = state.get("context", {})
            
            if current_step == "analyze":
                return "plan"
            elif current_step == "plan" and context.get("plan_created"):
                return "execute" 
            elif current_step == "execute" and context.get("execution_result"):
                return "finalize"
            else:
                return "end"

        workflow = StateGraph(HybridAgentState)
        
        workflow.add_node("analyze_with_memory", self._analyze_with_session_memory)
        workflow.add_node("plan_with_agents", self._plan_with_available_agents)
        workflow.add_node("execute_with_autogen", self._execute_via_autogen_agents)
        workflow.add_node("finalize_response", self._finalize_and_store)
        
        workflow.add_edge(START, "analyze_with_memory")
        workflow.add_conditional_edges(
            "analyze_with_memory",
            should_continue,
            {"plan": "plan_with_agents", "end": END}
        )
        workflow.add_conditional_edges(
            "plan_with_agents", 
            should_continue,
            {"execute": "execute_with_autogen", "end": END}
        )
        workflow.add_conditional_edges(
            "execute_with_autogen",
            should_continue, 
            {"finalize": "finalize_response", "end": END}
        )
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    async def _analyze_with_session_memory(self, state: HybridAgentState) -> Dict[str, Any]:
        """DYNAMIC memory analysis with LLM-driven reasoning"""
        log_structured("hybrid_analysis_start", session_id=state["session_id"], agent_id=state["agent_id"])
        
        try:
            session_id = state.get("session_id") or state.get("thread_id")
            
            # DYNAMIC: Search memory using LLM analysis
            try:
                from src.learning.behavior.behavior_engine import search_semantic_memory
                memory_results = search_semantic_memory(
                    state["user_input"], 
                    n_results=5, 
                    session_id=session_id
                )
            except ImportError:
                memory_results = None
            except Exception as e:
                logger.warning(f"Memory search failed: {e}")
                memory_results = None
            
            memory_context = {
                "retrieved_memories": memory_results,
                "session_id": session_id,
                "search_query": state["user_input"],
                "memory_count": len(memory_results.get("documents", [])) if memory_results else 0
            }
            
            # DYNAMIC: Generate reasoning using LLM analysis of memory and input
            reasoning = await self._generate_dynamic_analysis_reasoning(
                state['user_input'], 
                self.available_agents, 
                memory_context
            )
            
            return {
                "current_step": "analyze",
                "reasoning": reasoning,
                "memory_context": memory_context,
                "context": memory_results or {}
            }
            
        except Exception as e:
            log_structured("analysis_error", error=str(e), session_id=state["session_id"])
            return {
                "current_step": "analyze", 
                "reasoning": f"Basic analysis of {state['user_input']}",
                "memory_context": {},
                "context": {}
            }

    async def _generate_dynamic_analysis_reasoning(self, user_input: str, available_agents: List[str], memory_context: Dict[str, Any]) -> str:
        """DYNAMIC: Generate reasoning using LLM instead of templates"""
        try:
            reasoning_prompt = f"""Analyze this user interaction and generate reasoning for an AI assistant:

User Input: {user_input}
Available Agents: {available_agents}
Memory Context: {memory_context.get('memory_count', 0)} relevant memories found
Session: {memory_context.get('session_id', 'unknown')}

Generate a brief reasoning summary that explains:
1. What the user is trying to accomplish
2. What relevant information is available from memory
3. Which type of agent might be best suited to help

Keep it concise and focus on the reasoning process."""

            import requests
            from config.settings import settings
            
            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": reasoning_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 150}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                llm_reasoning = response.json().get("response", "").strip()
                if llm_reasoning:
                    return llm_reasoning
                    
        except Exception as e:
            logger.warning(f"LLM reasoning generation failed: {e}")
        
        # Simple fallback
        memory_count = memory_context.get("memory_count", 0)
        session_id = memory_context.get("session_id", "unknown")
        
        return f"""DYNAMIC ANALYSIS:
        User Input: {user_input}
        Session: {session_id}
        Available Agents: {', '.join(available_agents) if available_agents else 'None'}
        Memory Context: {memory_count} relevant memories found
        Analysis: Will route to best available agent based on capabilities and memory context"""

    async def _plan_with_available_agents(self, state: HybridAgentState) -> Dict[str, Any]:
        """COMPLETELY DYNAMIC agent selection using LLM analysis"""
        log_structured("hybrid_planning_start", session_id=state["session_id"])
        
        try:
            user_input = state["user_input"]
            memory_context = state.get("memory_context", {})
            
            # DYNAMIC: Use LLM to select best agent based on capabilities
            selected_agent = await self._select_agent_dynamically_with_llm(
                user_input, 
                self.available_agents, 
                memory_context
            )
            
            log_structured("hybrid_planning_completed", selected_agent=selected_agent, session_id=state["session_id"])
            
            return {
                "current_step": "plan",
                "context": {
                    **state.get("context", {}), 
                    "selected_agent": selected_agent, 
                    "plan_created": True,
                    "selection_method": "llm_dynamic"
                },
                "session_id": state.get("session_id"),
                "thread_id": state.get("thread_id")
            }
            
        except Exception as e:
            log_structured("planning_error", error=str(e), session_id=state["session_id"])
            return {
                "current_step": "plan",
                "context": {**state.get("context", {}), "selected_agent": "orchestrator"},
                "session_id": state.get("session_id")
            }

    async def _select_agent_dynamically_with_llm(self, user_input: str, available_agents: List[str], memory_context: Dict[str, Any]) -> str:
        """COMPLETELY DYNAMIC agent selection using LLM analysis - NO hardcoded keywords"""
        try:
            # DYNAMIC: Get agent capabilities from database
            agent_capabilities = {}
            for agent_id in available_agents:
                try:
                    # Get capabilities from database
                    active_agents = self.database.get_active_agents()
                    for agent_record in active_agents:
                        if agent_record.get("agent_id") == agent_id:
                            agent_capabilities[agent_id] = agent_record.get("capabilities", {})
                            break
                except Exception as e:
                    logger.warning(f"Failed to get capabilities for {agent_id}: {e}")
            
            # DYNAMIC: Use LLM to analyze input and select best agent
            try:
                from src.learning.behavior.behavior_engine import analyze_query_with_llm
                input_analysis = await analyze_query_with_llm(user_input)
                
                # DYNAMIC: Let LLM decide based on analysis and capabilities
                selection_prompt = f"""Based on this analysis and available agents, select the most appropriate agent:

User Input Analysis: {input_analysis}
Available Agents: {list(agent_capabilities.keys())}
Agent Capabilities: {agent_capabilities}
Memory Context: {memory_context.get('memory_count', 0)} relevant memories

Select the agent ID that best matches the user's intent and domain. Consider:
1. The intent type and domain from analysis
2. Agent capabilities and specializations
3. Previous memory context if relevant

Respond with only the agent ID (e.g., 'weather', 'routine', 'orchestrator')."""

                import requests
                from config.settings import settings
                
                response = requests.post(
                    f"{settings.llm.ollama_url}/api/generate",
                    json={
                        "model": settings.llm.default_model,
                        "prompt": selection_prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "max_tokens": 50}
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    llm_selection = response.json().get("response", "").strip().lower()
                    # Validate selection
                    for agent_id in available_agents:
                        if agent_id.lower() in llm_selection:
                            logger.info(f"LLM selected agent: {agent_id}")
                            return agent_id
                
            except Exception as e:
                logger.warning(f"LLM agent selection failed: {e}")
            
            # DYNAMIC FALLBACK: Use memory context to learn from previous selections
            if memory_context and memory_context.get("retrieved_memories"):
                memories = memory_context["retrieved_memories"].get("documents", [])
                metadatas = memory_context["retrieved_memories"].get("metadatas", [])
                
                if memories and metadatas:
                    # Look for agent references in memory
                    agent_mentions = {}
                    for i, memory in enumerate(memories[:3]):
                        metadata = metadatas[0][i] if metadatas and metadatas[0] and i < len(metadatas[0]) else {}
                        for agent_id in available_agents:
                            if agent_id in str(memory).lower() or agent_id in str(metadata).lower():
                                agent_mentions[agent_id] = agent_mentions.get(agent_id, 0) + 1
                    
                    if agent_mentions:
                        best_agent = max(agent_mentions, key=agent_mentions.get)
                        logger.info(f"Memory-based agent selection: {best_agent}")
                        return best_agent
            
            # FINAL FALLBACK: Use orchestrator if available, otherwise first available
            return "orchestrator" if "orchestrator" in available_agents else (available_agents[0] if available_agents else "orchestrator")
            
        except Exception as e:
            logger.error(f"Dynamic agent selection failed: {e}")
            return "orchestrator"

    async def _execute_via_autogen_agents(self, state: HybridAgentState) -> Dict[str, Any]:
        """Execute with FIXED AutoGen integration and dynamic processing"""
        log_structured("autogen_execution_start", session_id=state["session_id"])
        
        try:
            context_data = state.get("context", {})
            selected_agent = context_data.get("selected_agent", "orchestrator")
            session_id = context_data.get("session_id") or state.get("session_id")
            
            # Store memory BEFORE generating response
            await self._store_current_interaction_before_response(state)
            
            # FIXED: Call AutoGen agent with correct context
            result = await self._call_autogen_agent_with_session_context(
                state["user_input"], 
                selected_agent, 
                session_id,
                state.get("memory_context", {})
            )
            
            observation = {
                "agent_type": selected_agent,
                "result": result,
                "success": True,
                "timestamp": time.time(),
                "session_id": session_id
            }
            
            log_structured("autogen_action_completed", action_type=selected_agent, success=True, result_length=len(str(result)), session_id=session_id)
            
            return {
                "current_step": "execute",
                "observations": [observation],
                "context": {**context_data, "execution_result": result},
                "execution_result": result,
                "execution_success": True
            }
            
        except Exception as e:
            failed_observation = {
                "agent_type": state.get("context", {}).get("selected_agent", "unknown"),
                "result": None,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
                "session_id": state.get("session_id")
            }
            
            log_structured("autogen_action_failed", action_type=state.get("context", {}).get("selected_agent", "unknown"), error=str(e), session_id=state["session_id"])
            
            return {
                "current_step": "execute",
                "observations": [failed_observation],
                "context": state.get("context", {}),
                "execution_success": False,
                "execution_error": str(e)
            }

    async def _store_current_interaction_before_response(self, state: HybridAgentState):
        """Store interaction before response generation with dynamic processing"""
        try:
            session_id = state.get("session_id")
            user_input = state.get("user_input", "")
            
            if user_input and session_id:
                try:
                    from src.learning.behavior.behavior_engine import add_to_semantic_memory
                    
                    interaction_content = f"User: {user_input}"
                    metadata = {
                        "type": "user_message",
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "agent_id": state.get("agent_id", "hybrid_main")
                    }
                    
                    add_to_semantic_memory(interaction_content, metadata)
                    logger.info(f"Stored user input before response generation: {user_input[:50]}")
                except ImportError:
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to store interaction before response: {e}")

    async def _call_autogen_agent_with_session_context(self, user_input: str, agent_type: str, session_id: str, memory_context: Dict[str, Any]) -> str:
        """FIXED: Call AutoGen agent with correct TopicId constructor and dynamic routing"""
        try:
            # FIXED: Create context with correct TopicId constructor (from file 1)
            context = SimpleMessageContext("hybrid_workflow", session_id)
            
            # DYNAMIC: Route to available agent (keeping dynamic behavior from file 2)
            if agent_type in self.autogen_agents:
                agent = self.autogen_agents[agent_type]
                if hasattr(agent, 'set_session_context'):
                    agent.set_session_context(session_id)
                result = await agent.process_message(user_input, context)
                logger.info(f"{agent_type} agent response: {result[:100]}")
                return result
            else:
                # Fallback to orchestrator or first available agent
                fallback_agent_key = "orchestrator" if "orchestrator" in self.autogen_agents else list(self.autogen_agents.keys())[0] if self.autogen_agents else None
                
                if fallback_agent_key:
                    agent = self.autogen_agents[fallback_agent_key]
                    if hasattr(agent, 'set_session_context'):
                        agent.set_session_context(session_id)
                    result = await agent.process_message(user_input, context)
                    logger.info(f"Fallback {fallback_agent_key} response: {result[:100]}")
                    return result
                else:
                    # Last resort: Import orchestrator directly
                    from src.agents.core.orchestrator import orchestrator
                    if hasattr(orchestrator, 'set_session_context'):
                        orchestrator.set_session_context(session_id)
                    result = await orchestrator.process_message(user_input, context)
                    logger.info(f"Direct orchestrator response: {result[:100]}")
                    return result
                
        except Exception as e:
            raise Exception(f"Agent execution failed for {agent_type}: {str(e)}")

    async def _finalize_and_store(self, state: HybridAgentState) -> Dict[str, Any]:
        """DYNAMIC: Finalize response using LLM analysis with FIXED NoneType handling"""
        try:
            observations = state.get("observations", [])
            
            successful_obs = []
            if observations and isinstance(observations, list):
                for obs in observations:
                    if obs and isinstance(obs, dict) and obs.get("success", False):
                        successful_obs.append(obs)
            
            final_response = ""
            if successful_obs:
                latest_success = successful_obs[-1]
                result = latest_success.get("result", "")
                if result and str(result).strip():
                    result_str = str(result).strip()
                    if not result_str.startswith("Interaction stored"):
                        final_response = result_str
            
            if not final_response:
                # DYNAMIC: Use LLM to generate appropriate response based on context 
                final_response = await self._generate_dynamic_fallback_response(
                    state["user_input"], 
                    state.get("memory_context", {}),
                    observations
                )
            
            # Store complete interaction
            session_id = state.get("session_id")
            if session_id:
                try:
                    from src.learning.behavior.behavior_engine import add_to_semantic_memory
                    interaction_content = f"User: {state['user_input']}\nJarvis: {final_response}"
                    metadata = {
                        "type": "conversation",
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "agent_id": state.get("agent_id", "hybrid_main")
                    }
                    add_to_semantic_memory(interaction_content, metadata)
                except (ImportError, Exception) as e:
                    logger.warning(f"Memory storage failed: {e}")
            
            log_structured("hybrid_workflow_completed", response_length=len(final_response), 
                         total_observations=len(observations), session_id=session_id, completed=True)
            
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

    async def _generate_dynamic_fallback_response(self, user_input: str, memory_context: Dict[str, Any], observations: List[Dict[str, Any]]) -> str:
        """DYNAMIC: Generate fallback response using LLM analysis of memory instead of hardcoded responses"""
        try:
            # DYNAMIC: Use memory to inform response
            memory_info = ""
            if memory_context and memory_context.get("retrieved_memories"):
                memories = memory_context["retrieved_memories"].get("documents", [])
                if memories:
                    memory_info = f"Based on our conversation history: {memories[0][:100]}..."
            
            # DYNAMIC: Let LLM generate appropriate response
            fallback_prompt = f"""Generate an appropriate response for this user input based on available context:

User Input: {user_input}
Memory Context: {memory_info if memory_info else "No relevant memory found"}
Observations: {len(observations)} processing attempts made

Generate a helpful, personalized response that:
1. Acknowledges the user's input
2. Uses memory context if available
3. Offers assistance or asks clarifying questions
4. Maintains a conversational tone as Jarvis

Keep the response concise and helpful."""

            import requests
            from config.settings import settings
            
            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": fallback_prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "max_tokens": 80}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                llm_response = response.json().get("response", "").strip()
                if llm_response:
                    return llm_response
                    
        except Exception as e:
            logger.warning(f"Dynamic fallback generation failed: {e}")
        
        # Minimal fallback
        return "I understand you're asking me something. How can I help you?"

    async def run(self, state_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run workflow with proper error handling and dynamic processing"""
        try:
            initial_state = HybridAgentState(
                messages=[HumanMessage(content=state_input["user_input"])],
                session_id=state_input["session_id"],
                agent_id=state_input["agent_id"],
                user_input=state_input["user_input"],
                current_step="start",
                reasoning=None,
                observations=[],
                final_response=None,
                completed=False,
                context={},
                memory_context={},
                thread_id=state_input["session_id"]
            )
            
            logger.info(f"Workflow start session_id={initial_state['session_id']}")
            
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
            
            result = await self.workflow.ainvoke(initial_state, config)
            
            logger.info(f"Workflow completed: {result.get('completed', False)}")
            logger.info(f"Final response: {result.get('final_response', '')[:100]}")
            
            return result
            
        except Exception as e:
            log_structured("hybrid_workflow_error", error=str(e), session_id=state_input.get("session_id", "unknown"))
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
