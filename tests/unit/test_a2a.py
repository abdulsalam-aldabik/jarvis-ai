import asyncio
from agents.core.main_agent import JarvisMainAgent
from agents.core.database import DatabaseManager

async def test_a2a():
    try:
        db = DatabaseManager()
        await db.init_db()
        agent = JarvisMainAgent(database=db)
        
        # Test agent discovery
        agents = await agent.discover_agents()
        print(f'✅ A2A discovery works: Found {len(agents)} agents')
        
        # Test agent card
        card = await agent.get_agent_card()
        print(f"✅ Agent card works: {card['name']}")

        
    except Exception as e:
        print(f'❌ A2A test failed: {e}')

asyncio.run(test_a2a())