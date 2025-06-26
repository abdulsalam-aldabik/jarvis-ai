import asyncio
from agents.core.database import DatabaseManager

async def test_db():
    db = DatabaseManager()
    await db._initialize_pool()
    print('✅ Database connection successful')

asyncio.run(test_db())