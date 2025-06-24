FROM sparfenyuk/mcp-proxy:latest

# Alpine Linux system dependencies
RUN apk add --update --no-cache \
    git \
    curl \
    build-base \
    python3-dev \
    gcc \
    musl-dev \
    libffi-dev

# Install uv package manager
RUN python3 -m ensurepip && pip install --no-cache-dir uv

# Install all required Python packages for weather agent MCP
RUN uv pip install --system \
    fastmcp>=2.4.0 \
    python-dotenv>=1.0.0 \
    aiohttp>=3.9.0 \
    autogen-agentchat>=0.4.0 \
    autogen-core>=0.4.0 \
    requests>=2.31.0 \
    psycopg2-binary>=2.9.9 \
    pandas>=2.1.3 \
    numpy>=1.25.2

# Set environment variables
ENV PATH="/usr/local/bin:$PATH" \
    UV_PYTHON_PREFERENCE=only-system \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create app directory
WORKDIR /app

# Verify critical installations
RUN python3 -c "import fastmcp; print('FastMCP installed successfully')" && \
    python3 -c "import autogen_core; print('AutoGen Core installed successfully')" && \
    python3 -c "import aiohttp; print('aiohttp installed successfully')"

# Copy MCP server files
COPY api/ ./api/

EXPOSE 8180

ENTRYPOINT ["mcp-proxy"]
