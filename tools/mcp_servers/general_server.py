#!/usr/bin/env python3
"""
General MCP Server for AutoGen 0.6.2 Integration
Compatible with sparfenyuk/mcp-proxy
"""
import asyncio
import json
import sys
import logging
from typing import Any, Dict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("general-server")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools for AutoGen 0.6.2"""
    logger.info("Tools requested by AutoGen system")
    return [
        Tool(
            name="get_weather",
            description="Get current weather information for any location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for (city, country)"
                    }
                },
                "required": ["location"]
            }
        ),
        Tool(
            name="create_routine",
            description="Create a new daily routine with specified activities",
            inputSchema={
                "type": "object", 
                "properties": {
                    "routine_data": {
                        "type": "object",
                        "description": "Routine configuration with name, type, and activities",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "activities": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["routine_data"]
            }
        ),
        Tool(
            name="list_routines",
            description="List all existing routines for a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier to filter routines"
                    }
                }
            }
        ),
        Tool(
            name="system_status",
            description="Get comprehensive system status and health information",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed component information"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from AutoGen 0.6.2 agents"""
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "get_weather":
            location = arguments.get("location", "Brussels")
            # Enhanced weather response
            result = {
                "location": location,
                "temperature": "19°C",
                "feels_like": "21°C", 
                "condition": "Partly cloudy",
                "humidity": "60%",
                "wind_speed": "13 km/h",
                "wind_direction": "SW",
                "visibility": "10 km",
                "uv_index": 3,
                "timestamp": "2025-07-08T03:44:00Z",
                "source": "AutoGen 0.6.2 MCP Weather Service"
            }
            
        elif name == "create_routine":
            routine_data = arguments.get("routine_data", {})
            routine_name = routine_data.get("name", "Unnamed Routine")
            routine_type = routine_data.get("type", "general")
            activities = routine_data.get("activities", [])
            
            result = {
                "status": "success",
                "routine_id": f"routine_{hash(routine_name) % 10000:04d}",
                "name": routine_name,
                "type": routine_type,
                "activities_count": len(activities),
                "message": f"Successfully created '{routine_name}' routine with {len(activities)} activities",
                "created_at": "2025-07-08T03:44:00Z"
            }
            
        elif name == "list_routines":
            session_id = arguments.get("session_id", "default")
            result = {
                "session_id": session_id,
                "routines": [
                    {
                        "id": "routine_0001", 
                        "name": "Morning Energizer", 
                        "type": "morning",
                        "activities": 6,
                        "last_completed": "2025-07-07"
                    },
                    {
                        "id": "routine_0002",
                        "name": "Evening Wind-Down",
                        "type": "evening", 
                        "activities": 5,
                        "last_completed": "2025-07-06"
                    },
                    {
                        "id": "routine_0003",
                        "name": "Workout Session",
                        "type": "exercise",
                        "activities": 4,
                        "last_completed": "2025-07-05"
                    }
                ],
                "total_count": 3
            }
            
        elif name == "system_status":
            include_details = arguments.get("include_details", False)
            result = {
                "system_status": "healthy",
                "autogen_version": "0.6.2-official",
                "mcp_server": "general-server",
                "uptime": "operational",
                "components": {
                    "weather_service": "active",
                    "routine_manager": "active", 
                    "tool_discovery": "active"
                }
            }
            
            if include_details:
                result["details"] = {
                    "available_tools": 4,
                    "successful_calls": "multiple",
                    "integration_status": "AutoGen 0.6.2 compatible"
                }
        
        else:
            result = {
                "error": f"Unknown tool: {name}",
                "available_tools": ["get_weather", "create_routine", "list_routines", "system_status"]
            }
            
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        error_result = {
            "error": f"Tool execution failed: {str(e)}",
            "tool": name,
            "arguments": arguments
        }
        return [TextContent(type="text", text=json.dumps(error_result))]

async def main():
    """Run the MCP server with proper error handling"""
    logger.info("Starting General MCP Server for AutoGen 0.6.2")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server connected via stdio")
            await server.run(
                read_stream, 
                write_stream, 
                server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"MCP Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
