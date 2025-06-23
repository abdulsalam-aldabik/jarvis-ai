
from prometheus_client import start_http_server, Counter

REQUESTS = Counter('agent_requests_total', 'Total requests to the agent')

def start_metrics_server():
    start_http_server(8001)  # Exposes metrics at :8001/metrics
