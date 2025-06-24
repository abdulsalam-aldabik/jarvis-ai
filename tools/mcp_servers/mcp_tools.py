import time
import signal
import sys

def signal_handler(sig, frame):
    print("Weather MCP shutting down gracefully...")
    sys.exit(0)

def get_weather():
    return "Weather service placeholder"

def main():
    print("Weather MCP service starting...")
    print(get_weather())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:
            print("☁️ Weather MCP service running...")
            time.sleep(30)  # 30-second heartbeat
    except KeyboardInterrupt:
        print("Weather MCP interrupted, shutting down...")

if __name__ == "__main__":
    main()
