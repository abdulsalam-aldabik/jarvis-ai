FROM sparfenyuk/mcp-proxy:latest

# Alpine Linux uses apk, not apt-get
RUN apk add --update --no-cache \
    git \
    curl \
    build-base \
    python3-dev

# Install uv using the method from search results
RUN python3 -m ensurepip && pip install --no-cache-dir uv

# Install FastMCP using uv (as recommended)
RUN uv pip install --system fastmcp>=2.4.0 python-dotenv>=1.0.0 aiohttp>=3.9.0

# Set environment variables as shown in search results
ENV PATH="/usr/local/bin:$PATH" \
    UV_PYTHON_PREFERENCE=only-system \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Verify installation
RUN python3 -c "import fastmcp; print('FastMCP installed successfully')"

ENTRYPOINT ["mcp-proxy"]
