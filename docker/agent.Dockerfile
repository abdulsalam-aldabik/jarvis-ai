FROM python:3.12-slim
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project structure (not just agent/)
COPY . .

ENV PYTHONPATH=/app

# Use the serve command for Docker containers
CMD ["python", "-m", "agent.main", "serve"]
