# FROM python:3.11-alpine

# # Install Node.js and npm
# RUN apk add --no-cache nodejs npm git build-base

# # Install Python packages (no virtual env needed in python base image)
# RUN pip install --no-cache-dir \
#     fastmcp>=2.4.0 \
#     aiohttp>=3.9.0 \
#     python-dotenv>=1.0.0 \
#     requests>=2.31.0

# # Clone the repository
# RUN git clone https://github.com/adamwattis/mcp-proxy-server.git /opt/mcp-proxy-server

# WORKDIR /opt/mcp-proxy-server

# # Install Node.js dependencies and build the server
# RUN npm install && npm run build

# # Create the API directory for Python scripts
# RUN mkdir -p /app/api

# # Copy your Python MCP servers
# COPY api/ /app/api/

# # Copy your config into the image
# COPY mcp-config.json /opt/mcp-proxy-server/config.json

# # Make Python scripts executable
# RUN chmod +x /app/api/*.py

# EXPOSE 8180

# # Set working directory and start the server
# WORKDIR /opt/mcp-proxy-server
# CMD ["sh", "-c", "MCP_CONFIG_PATH=/opt/mcp-proxy-server/config.json node build/index.js --port 8180"]


FROM ghcr.io/tbxark/mcp-proxy:latest

# Copy your config
COPY mcp-config.json /config/config.json
COPY api/ /app/api/

EXPOSE 9190

CMD ["--config", "/config/config.json"]
