# Stage 1: Builder using the *same* base image as the final stage
FROM ghcr.io/sparfenyuk/mcp-proxy:latest AS builder

# This proxy image is based on Alpine, so we use 'apk' to install build tools
RUN apk add --no-cache \
    build-base \
    gcc \
    musl-dev \
    libffi-dev \
    cargo \
    rust

# Create and activate a virtual environment
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies within the venv
COPY config/proxy-requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r proxy-requirements.txt

# Stage 2: Final lean runtime image
FROM ghcr.io/sparfenyuk/mcp-proxy:latest

# Copy the entire, fully compatible virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Verify the installation works in the final image
RUN python -c "import requests; import mcp; import rpds; print('✅ Dependencies including Rust-based rpds-py are correctly installed and compatible!')"
