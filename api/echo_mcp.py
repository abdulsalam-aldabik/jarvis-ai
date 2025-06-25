#!/usr/bin/env python3

from fastmcp import FastMCP

mcp = FastMCP("Echo", port=8181, debug=True , log_level="DEBUG" , host="0.0.0.0")


@mcp.tool()
async def echo(text: str) -> str:
    """Echoes the input text back."""
    return f"ECHO: {text}"

@mcp.tool()
async def add(a: float, b: float) -> float:
    """Returns the sum of two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="sse")  # for echo
