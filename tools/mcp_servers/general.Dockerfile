FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY tools/mcp_servers/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY tools/mcp_servers/general_server.py .

# Create data directory
RUN mkdir -p /app/data

# Health check for stdio-based MCP server
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD pgrep -f "python.*general_server.py" || exit 1

# Run server with proper signal handling
CMD ["python", "-u", "general_server.py"]
