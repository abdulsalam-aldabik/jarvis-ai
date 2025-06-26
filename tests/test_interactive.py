import asyncio
from agents.core.main_agent import JarvisMainAgent
from agents.core.database import DatabaseManager

async def interactive_test():
    db = DatabaseManager()
    await db._initialize_pool()
    agent = JarvisMainAgent(database=db)
    
    test_messages = [
        'Hello, test agentic mode',
        'What can you help me with?',
        'Tell me about the weather',
        'Remember that I like sunny days'
    ]
    
    for msg in test_messages:
        print(f'User: {msg}')
        try:
            if hasattr(agent, 'agentic_chat'):
                response = await agent.agentic_chat(msg)
            else:
                response = await agent.chat(msg)
            print(f'Jarvis: {response[:100]}...')
            print('---')
        except Exception as e:
            print(f'Error: {e}')
            print('---')

asyncio.run(interactive_test())