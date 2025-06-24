-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Agent state for configuration and persistent data
CREATE TABLE IF NOT EXISTS agent_state (
    id SERIAL PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Structured logging
CREATE TABLE IF NOT EXISTS agent_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    meta JSONB
);

-- Vector embeddings for semantic search
CREATE TABLE IF NOT EXISTS semantic_memory (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(1536), -- OpenAI embedding dimension
    content TEXT NOT NULL,
    meta JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Routine management
CREATE TABLE IF NOT EXISTS routines (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    schedule_time TIME,
    duration_minutes INTEGER,
    activities JSONB,
    approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat conversation history
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID DEFAULT gen_random_uuid(),
    user_message TEXT NOT NULL,
    agent_response TEXT,
    model_used TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_agent_logs_level ON agent_logs(level);
CREATE INDEX IF NOT EXISTS idx_agent_logs_timestamp ON agent_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_state_key ON agent_state(key);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_created_at ON semantic_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routines_approved ON routines(approved);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);

-- Vector similarity search index
CREATE INDEX IF NOT EXISTS idx_semantic_memory_embedding ON semantic_memory 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Initial configuration
INSERT INTO agent_state (key, value) VALUES 
    ('agent_version', '"1.0.0"'),
    ('initialized_at', to_jsonb(CURRENT_TIMESTAMP))
ON CONFLICT (key) DO NOTHING;

-- Initial log entry
INSERT INTO agent_logs (level, message, meta) VALUES 
    ('INFO', 'Database schema initialized', '{"source": "init_script", "version": "1.0"}');
