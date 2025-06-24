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
-- Agent interactions for multi-agent communication
CREATE TABLE IF NOT EXISTS agent_interactions (
    id SERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    interaction_type TEXT NOT NULL, -- 'request', 'response', 'notification', 'collaboration'
    data JSONB NOT NULL,
    parent_interaction_id INTEGER REFERENCES agent_interactions(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent registry for tracking active agents
CREATE TABLE IF NOT EXISTS agent_registry (
    id SERIAL PRIMARY KEY,
    agent_id TEXT UNIQUE NOT NULL,
    agent_type TEXT NOT NULL, -- 'weather', 'routine', 'device', 'orchestrator'
    capabilities JSONB NOT NULL,
    status TEXT DEFAULT 'active', -- 'active', 'inactive', 'error'
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Enhanced routines with execution tracking
ALTER TABLE routines ADD COLUMN IF NOT EXISTS execution_count INTEGER DEFAULT 0;
ALTER TABLE routines ADD COLUMN IF NOT EXISTS last_executed TIMESTAMP;
ALTER TABLE routines ADD COLUMN IF NOT EXISTS success_rate DECIMAL(5,2) DEFAULT 100.0;
ALTER TABLE routines ADD COLUMN IF NOT EXISTS learned_from TEXT; -- 'user_pattern', 'time_series', 'rule_learning'

-- Add agent_id to existing logs table for multi-agent support
ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS agent_id TEXT;

-- Performance indexes for multi-agent queries
CREATE INDEX IF NOT EXISTS idx_agent_interactions_agent_id ON agent_interactions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_type ON agent_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_created_at ON agent_interactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_registry_type ON agent_registry(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_registry_status ON agent_registry(status);
CREATE INDEX IF NOT EXISTS idx_agent_registry_heartbeat ON agent_registry(last_heartbeat DESC);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_id ON agent_logs(agent_id);

-- Insert initial agent registry entries
INSERT INTO agent_registry (agent_id, agent_type, capabilities) VALUES 
    ('main_orchestrator', 'orchestrator', '{"roles": ["coordination", "user_interface", "decision_making"]}'),
    ('weather_agent', 'weather', '{"roles": ["weather_data", "forecasting", "location_services"]}'),
    ('routine_agent', 'routine', '{"roles": ["routine_management", "scheduling", "habit_tracking"]}')
ON CONFLICT (agent_id) DO NOTHING;

-- Add configuration for multi-agent system
INSERT INTO agent_state (key, value) VALUES 
    ('multi_agent_enabled', 'true'),
    ('max_agents', '10'),
    ('agent_communication_timeout', '30')
ON CONFLICT (key) DO NOTHING;
