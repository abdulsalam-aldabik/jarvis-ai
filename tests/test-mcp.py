#!/usr/bin/env python3
"""Complete MCP system testing"""

import asyncio
import json
import httpx
import time

async def test_mcp_system():
    """Test complete MCP system"""
    print("🚀 Testing Complete MCP System...")
    
    # Test individual servers
    print("\n1. Testing Individual Servers...")
    
    async with httpx.AsyncClient() as client:
        # Test general server
        try:
            response = await client.get("http://localhost:8183/health")
            if response.status_code == 200:
                print("✅ General MCP Server: Healthy")
                health_data = response.json()
                print(f"   - Tools: {health_data.get('tools_available', [])}")
            else:
                print(f"❌ General MCP Server: {response.status_code}")
        except Exception as e:
            print(f"❌ General MCP Server: {e}")
        
        # Test weather server
        try:
            response = await client.get("http://localhost:8182/health")
            if response.status_code == 200:
                print("✅ Weather MCP Server: Healthy") 
                health_data = response.json()
                print(f"   - Mode: {'Demo' if health_data.get('demo_mode') else 'Real API'}")
            else:
                print(f"❌ Weather MCP Server: {response.status_code}")
        except Exception as e:
            print(f"❌ Weather MCP Server: {e}")
        
        # Test proxy
        try:
            response = await client.get("http://localhost:9190/")
            if response.status_code == 200:
                print("✅ MCP Proxy: Healthy")
            else:
                print(f"❌ MCP Proxy: {response.status_code}")
        except Exception as e:
            print(f"❌ MCP Proxy: {e}")
    
    print("\n2. Testing Tool Execution...")
    
    # Test tool execution via direct HTTP calls
    async with httpx.AsyncClient() as client:
        # Test general tools
        print("\n   General Tools:")
        
        # System info
        try:
            response = await client.post(
                "http://localhost:8183/tools/call",
                json={"name": "get_system_info", "arguments": {}}
            )
            if response.status_code == 200:
                print("   ✅ get_system_info: Success")
            else:
                print(f"   ❌ get_system_info: {response.status_code}")
        except Exception as e:
            print(f"   ❌ get_system_info: {e}")
        
        # Command execution
        try:
            response = await client.post(
                "http://localhost:8183/tools/call",
                json={"name": "execute_command", "arguments": {"command": "uptime"}}
            )
            if response.status_code == 200:
                print("   ✅ execute_command: Success")
            else:
                print(f"   ❌ execute_command: {response.status_code}")
        except Exception as e:
            print(f"   ❌ execute_command: {e}")
        
        # Weather tools
        print("\n   Weather Tools:")
        
        # Current weather
        try:
            response = await client.post(
                "http://localhost:8182/tools/call",
                json={"name": "get_weather", "arguments": {"location": "Brussels"}}
            )
            if response.status_code == 200:
                print("   ✅ get_weather: Success")
            else:
                print(f"   ❌ get_weather: {response.status_code}")
        except Exception as e:
            print(f"   ❌ get_weather: {e}")
        
        # Weather forecast
        try:
            response = await client.post(
                "http://localhost:8182/tools/call",
                json={"name": "get_forecast", "arguments": {"location": "Brussels", "days": 3}}
            )
            if response.status_code == 200:
                print("   ✅ get_forecast: Success")
            else:
                print(f"   ❌ get_forecast: {response.status_code}")
        except Exception as e:
            print(f"   ❌ get_forecast: {e}")
    
    print("\n🎉 Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_mcp_system())
