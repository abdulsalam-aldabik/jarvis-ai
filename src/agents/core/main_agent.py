import asyncio
import uuid
import logging
from typing import Dict, Any, Optional

import typer
from autogen_agentchat.ui import Console

from src.agents.core.groupchat_manager import groupchat_manager
from src.agents.core.mcp_tools import mcp_tools_manager
from src.agents.core.logging_config import log_structured
from config.settings import settings

logger = logging.getLogger("jarvis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class MainAgent:
    """Main AutoGen 0.6.2 application with GroupChatManager and CLI"""
    
    instance = None
    
    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.groupchat = groupchat_manager
        # ✅ Remove Console instantiation - use it directly in methods
        self.initialized = True
        
        logger.info("AutoGen 0.6.2 Main Application initialized")

    async def initialize_system(self):
        """Initialize AutoGen 0.6.2 system components"""
        try:
            # Initialize MCP tools
            await mcp_tools_manager.initialize_mcp_tools()
            # Initialize GroupChat agents
            self.groupchat.initialize_agents()
            
            
            
            log_structured("autogen_062_system_initialized",
                         autogen_version="0.6.2-official")
                         
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    async def process_message_stream(self, message: str, session_id: str = None):
        """Process message using AutoGen 0.6.2 with Console streaming"""
        session_id = session_id or str(uuid.uuid4())
        
        try:
            # Get the agent stream and use Console properly
            agent_stream = await self.groupchat.run_stream(task=message, session_id=session_id)
            
            # ✅ Correct Console usage with stream parameter
            await Console(agent_stream)
            
            log_structured("autogen_062_main_success",
                         session_id=session_id)
                         
        except Exception as e:
            logger.error(f"AutoGen 0.6.2 streaming failed: {e}")
            print(f"❌ Error: {str(e)}")

    async def process_message(self, message: str, session_id: str = None) -> str:
        """Process message using AutoGen 0.6.2 GroupChatManager (non-streaming)"""
        session_id = session_id or str(uuid.uuid4())
        
        try:
            response = await self.groupchat.process_message(message, session_id)
            
            log_structured("autogen_062_main_success",
                         session_id=session_id,
                         response_length=len(response))
                         
            return response
            
        except Exception as e:
            logger.error(f"AutoGen 0.6.2 processing failed: {e}")
            return f"I encountered an error: {str(e)}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return self.groupchat.get_system_status()


# Global main agent instance
main_agent = MainAgent()

# Typer CLI Application
app = typer.Typer()

@app.command()
def chat():
    """Interactive chat using AutoGen 0.6.2 GroupChatManager"""
    print("🤖 Jarvis AutoGen 0.6.2 GroupChat")
    print("Type 'quit' to exit, 'status' for system status, 'stream' for streaming mode")
    
    async def chat_loop():
        agent = main_agent
        await agent.initialize_system()
        session_id = str(uuid.uuid4())
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ["quit", "exit"]:
                    break
                elif user_input.lower() == "status":
                    status = agent.get_system_status()
                    print(f"📊 System: {status['system_type']}")
                    print(f"🔧 AutoGen: {status['autogen_version']}")
                    print(f"👥 Agents: {status['agents']['count']}")
                    continue
                elif user_input.lower() == "stream":
                    print("🌊 Switching to streaming mode...")
                    stream_input = input("💬 Stream message: ").strip()
                    if stream_input:
                        await agent.process_message_stream(stream_input, session_id)
                    continue
                
                if user_input:
                    print("🤔 Processing with AutoGen 0.6.2 GroupChat...", end="", flush=True)
                    response = await agent.process_message(user_input, session_id)
                    print(f"\n🤖 Jarvis: {response}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    asyncio.run(chat_loop())

@app.command()
def status():
    """Check AutoGen 0.6.2 system status"""
    try:
        print("🔍 AutoGen 0.6.2 System Status Check")
        print("-" * 40)
        
        # Check imports
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_ext.models.ollama import OllamaChatCompletionClient
            from autogen_ext.tools.mcp import mcp_server_tools
            print("✅ AutoGen 0.6.2 imports: SUCCESS")
        except ImportError as e:
            print(f"❌ AutoGen 0.6.2 imports: FAILED - {e}")
            return
        
        # Check system status
        status = main_agent.get_system_status()
        print(f"📊 System Type: {status['system_type']}")
        print(f"🔧 AutoGen Version: {status['autogen_version']}")
        print(f"👥 Agents: {status['agents']['count']}")
        print(f"🔄 GroupChat: {'✅' if status['groupchat_initialized'] else '❌'}")
        
        print("\n🎉 AutoGen 0.6.2 migration successful!")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")

if __name__ == "__main__":
    app()
