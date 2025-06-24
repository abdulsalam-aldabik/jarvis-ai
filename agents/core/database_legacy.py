import os
import psycopg2
import requests

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://jarvis:strongpassword@postgres:5432/jarvisdb")
CHROMA_URL = os.getenv("CHROMA_URL", "http://chroma:8000")

def get_pg_conn():
    return psycopg2.connect(POSTGRES_URL)

def log_event(level, message, meta=None):
    conn = get_pg_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_logs (level, message, meta) VALUES (%s, %s, %s)",
            (level, message, meta)
        )
    conn.close()

def add_semantic_memory(embedding, content, meta=None):
    # Example call to Chroma REST API (expand as needed)
    r = requests.post(f"{CHROMA_URL}/api/v1/collections/default/documents", json={
        "embedding": embedding,
        "content": content,
        "meta": meta or {}
    })
    return r.ok
