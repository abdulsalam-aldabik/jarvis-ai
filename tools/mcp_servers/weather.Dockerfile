# tools/mcp_servers/weather.Dockerfile - FIXED FASTMCP
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# FIXED: Install correct FastMCP library
RUN pip install --no-cache-dir fastmcp aiohttp python-dotenv pyyaml

# Copy weather service code
COPY ./api/mcp_weather.py .
COPY config/.env* ./

# Create cache directory
RUN mkdir -p /app/.cache/weather

EXPOSE 8182

# Health check using SSE endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f -m 5 http://localhost:8182/sse/ | head -n1 | grep -q "event" || exit 1

ENV PYTHONPATH=/app

# Run FastMCP server
CMD ["python", "mcp_weather.py"]
