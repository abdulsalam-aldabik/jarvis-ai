FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api/ ./api/
COPY .env* ./

CMD ["python", "-m", "api.weather_mcp"]
