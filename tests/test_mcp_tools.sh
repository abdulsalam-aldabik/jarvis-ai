#!/bin/bash
# test_mcp_tools.sh

echo "=== Getting MCP Session Endpoint ==="
SESSION_RESPONSE=$(curl -s http://localhost:9190/weather/sse | head -5)
echo "$SESSION_RESPONSE"

# Extract session ID (you might need to adjust this)
SESSION_ID=$(echo "$SESSION_RESPONSE" | grep "sessionId=" | sed 's/.*sessionId=\([^"]*\).*/\1/')

if [ -z "$SESSION_ID" ]; then
    echo "Could not extract session ID"
    exit 1
fi

echo -e "\n=== Using Session ID: $SESSION_ID ==="

ENDPOINT="http://localhost:9190/weather/message?sessionId=$SESSION_ID"

echo -e "\n=== Testing get_weather ==="
curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1", 
    "method": "tools/call",
    "params": {
      "name": "get_weather",
      "arguments": {"location": "Brussels"}
    }
  }' | jq .

echo -e "\n=== Testing get_forecast ==="
curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tools/call", 
    "params": {
      "name": "get_forecast",
      "arguments": {"location": "Paris", "days": 5}
    }
  }' | jq .
