"""
Test script for agentic workflow
"""
import asyncio
from agents.core.database import DatabaseManager  # Changed from Database
from agents.core.agentic_state import ReasoningState
from agents.core.agentic_workflow import AgenticWorkflow

async def test_agentic_workflow():
    """Test the agentic workflow"""
    print("🧪 Testing Agentic Workflow...")
    
    # Initialize database
    database = DatabaseManager()  # Changed from Database()
    
    # Create workflow
    workflow = AgenticWorkflow(database)
    
    # Create test state
    state = ReasoningState(
        agent_id="test_agent",
        user_input="What's the weather like today?"
    )
    
    # Run workflow
    result = await workflow.run(state)
    
    print(f"✅ Workflow completed!")
    print(f"📝 Reasoning: {result.reasoning}")
    print(f"📋 Plan: {len(result.plan)} actions")
    print(f"👀 Observations: {len(result.observations)}")
    print(f"💬 Response: {result.final_response}")

if __name__ == "__main__":
    asyncio.run(test_agentic_workflow())
