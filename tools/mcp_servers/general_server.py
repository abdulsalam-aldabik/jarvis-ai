#!/usr/bin/env python3
""" Stdio-based General MCP server compatible with sparfenyuk/mcp-proxy """
import asyncio, json, os, subprocess, logging
from datetime import datetime
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Tool, TextContent, CallToolResult
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
server = Server("general-tools")
AGENT_ID = "mcp_general_server"

async def get_system_info():
    return json.dumps({
        "timestamp": datetime.now().isoformat(), "agent_id": AGENT_ID,
        "cwd": os.getcwd(), "user": os.getenv("USER", "unknown")
    })

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name="get_system_info", description="Get system info", inputSchema={})]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_system_info":
        result = await get_system_info()
        return CallToolResult(content=[TextContent(type="text", text=result)])
    return CallToolResult(content=[TextContent(type="text", text='{"error":"Unknown tool"}')])

async def main():
    logging.info("Starting stdio general server")
    async with stdio_server() as (reader, writer):
        # FIXED: Add required server_version and capabilities fields
        capabilities = server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={}
        )
        
        init_options = InitializationOptions(
            server_name="general-tools",
            server_version="1.0.0", 
            capabilities=capabilities
        )
        
        await server.run(reader, writer, init_options)
    logging.info("General server stopped.")

if __name__ == "__main__":
    asyncio.run(main())
