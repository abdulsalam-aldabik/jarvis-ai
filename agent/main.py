import typer
import time
import signal
import sys
from typing import Optional

app = typer.Typer()

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("🛑 Jarvis AI Agent shutting down gracefully...")
    running = False

@app.command()
def hello():
    """Say hello from Jarvis AI."""
    print("Hello from Jarvis-AI agent!")

@app.command()
def status():
    """Check agent status."""
    print("🤖 Jarvis AI Agent is operational")
    print("📊 All systems running normally")

@app.command()
def serve():
    """Run Jarvis AI agent as a service (for Docker)."""
    global running
    
    print("🚀 Starting Jarvis AI Agent service...")
    print("🤖 Agent is ready and monitoring...")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while running:
            # Your AI agent logic here
            print("💚 Agent heartbeat - all systems operational")
            time.sleep(30)  # 30-second heartbeat
    except KeyboardInterrupt:
        print("Agent interrupted, shutting down...")
    
    print("✅ Jarvis AI Agent service stopped")

@app.command()
def chat(message: str):
    """Chat with Jarvis AI."""
    print(f"🤖 Jarvis: Processing your message: '{message}'")
    # Add your AI processing logic here
    print("🤖 Jarvis: I'm still learning! More capabilities coming soon.")

if __name__ == "__main__":
    app()


from fastapi import FastAPI
from fastapi_health import health

@app.get("/health")
def healthy():
    return True

app = FastAPI()
app.add_api_route("/health", health([healthy]))

