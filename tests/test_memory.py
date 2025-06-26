import asyncio
from agents.core.database import DatabaseManager

async def test_memory():
    try:
        db = Database()
        await db.init_db()
        
        # Store an interaction
        await db.store_interaction(
            agent_id='test_agent',
            user_input='Test memory storage',
            agent_response='Memory stored successfully',
            metadata={'test': True}
        )
        print('✅ Memory storage works')
        
        # Search memories
        memories = await db.search_memories('memory storage', limit=3)
        print(f'✅ Memory search works: Found {len(memories)} memories')
        
    except Exception as e:
        print(f'❌ Memory test failed: {e}')

asyncio.run(test_memory())