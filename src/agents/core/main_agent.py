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
        """Process message using AutoGen 0.6.2 with proper streaming implementation"""
        session_id = session_id or str(uuid.uuid4())
        
        try:
            agent_stream = await self.groupchat.run_stream(task=message, session_id=session_id)
            
            print("🤖 Jarvis: ", end="", flush=True)
            
            full_response = ""
            async for chunk in agent_stream:
                if isinstance(chunk, dict):
                    # Handle structured streaming response
                    content = chunk.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
                elif isinstance(chunk, str):
                    # Handle simple string chunks
                    print(chunk, end="", flush=True)
                    full_response += chunk
                else:
                    # Handle unexpected chunk types
                    chunk_str = str(chunk)
                    print(chunk_str, end="", flush=True)
                    full_response += chunk_str
            
            # Add newline after streaming is complete
            print()
            
            log_structured("autogen_062_streaming_success",
                         session_id=session_id,
                         response_length=len(full_response),
                         streaming_enabled=True)
                         
        except Exception as e:
            logger.error(f"AutoGen 0.6.2 streaming failed: {e}")
            print(f"❌ Streaming Error: {str(e)}")
            log_structured("autogen_062_streaming_failed", 
                         session_id=session_id, 
                         error=str(e))

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

    async def interactive_stream_mode(self, session_id: str):
        """Enhanced interactive streaming mode with proper error handling"""
        print("🌊 **Streaming Mode Activated**")
        print("Type your messages and see responses stream in real-time")
        print("Commands: 'back' to return to normal mode, 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Stream: ").strip()
                
                if user_input.lower() in ["back", "normal"]:
                    print("🔄 Returning to normal mode...")
                    break
                elif user_input.lower() in ["quit", "exit"]:
                    return "quit"
                elif user_input:
                    await self.process_message_stream(user_input, session_id)
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to normal mode...")
                break
            except Exception as e:
                print(f"\n❌ Stream Error: {e}")
                logger.error(f"Interactive streaming error: {e}")
        
        return "continue"



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
                    print(f"🧠 Intelligence: {', '.join(status.get('intelligence_features', []))}")
                    continue
                elif user_input.lower() == "stream":
                    result = await agent.interactive_stream_mode(session_id)
                    if result == "quit":
                        break
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
def stream():
    """Direct streaming mode for testing"""
    print("🌊 **Direct Streaming Mode**")
    print("Enter a message to see streaming response:")
    
    async def stream_test():
        agent = main_agent
        await agent.initialize_system()
        
        message = input("💬 Message: ").strip()
        if message:
            await agent.process_message_stream(message)
    
    asyncio.run(stream_test())

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
        
        # Check intelligence features
        if 'intelligence_features' in status:
            print("🧠 Intelligence Features:")
            for feature in status['intelligence_features']:
                print(f"   • {feature}")
        
        print("\n🎉 AutoGen 0.6.2 with Intelligence Edition: OPERATIONAL!")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")

if __name__ == "__main__":
    app()
