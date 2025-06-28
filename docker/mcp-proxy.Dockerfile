FROM ghcr.io/tbxark/mcp-proxy:latest

# Copy your config
COPY  config/mcp-config.json /config/config.json
COPY api/ /app/api/

EXPOSE 9190

CMD ["--config", "/config/config.json"]
