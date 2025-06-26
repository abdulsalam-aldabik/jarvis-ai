import asyncio
from agents.core.main_agent import JarvisMainAgent
from agents.core.database import DatabaseManager

async def test_chat():
    try:
        db = Database()
        await db.init_db()
        agent = JarvisMainAgent(database=db)
        response = await agent.chat('Hello, how are you?')
        print(f'✅ Basic chat works: {response[:50]}...')
    except Exception as e:
        print(f'❌ Basic chat failed: {e}')

asyncio.run(test_chat())