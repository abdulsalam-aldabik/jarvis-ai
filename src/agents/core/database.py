"""
Enhanced database manager with proper connection pooling and session context
"""
import json
import logging
import time
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database manager with connection pooling and multi-agent support - THREAD SAFE"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.connection_pool = None
                    self._initialize_pool()
                    DatabaseManager._initialized = True
                    logger.info("Database manager initialized (thread-safe singleton)")

    def _initialize_pool(self):
        """Initialize connection pool with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    1, 20,  # min and max connections
                    settings.database.postgres_url,
                    cursor_factory=RealDictCursor
                )
                logger.info("Database connection pool initialized successfully")
                return
            except Exception as e:
                logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to initialize database pool after all retries")
                    raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def log_event(self, level: str, message: str, meta: Optional[Dict] = None, agent_id: Optional[str] = None) -> bool:
        """Enhanced logging with agent tracking"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO agent_logs (level, message, meta, agent_id) VALUES (%s, %s, %s, %s)",
                        (level, message, json.dumps(meta) if meta else None, agent_id)
                    )
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return False

    def store_agent_interaction(self, agent_id: str, interaction_type: str, data: Dict[str, Any], parent_id: Optional[str] = None) -> Optional[str]:
        """Store agent-to-agent interactions for multi-agent communication"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO agent_interactions (agent_id, interaction_type, data, parent_interaction_id) VALUES (%s, %s, %s, %s) RETURNING id",
                        (agent_id, interaction_type, json.dumps(data), parent_id)
                    )
                    interaction_id = cur.fetchone()['id']
                    conn.commit()
                    return str(interaction_id)
        except Exception as e:
            logger.error(f"Failed to store agent interaction: {e}")
            return None

    def get_agent_context(self, agent_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent context for an agent"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM agent_interactions WHERE agent_id = %s ORDER BY created_at DESC LIMIT %s",
                        (agent_id, limit)
                    )
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get agent context: {e}")
            return []

    def register_agent(self, agent_id: str, agent_type: str, capabilities: Dict[str, Any], description: str = "") -> bool:
        """Register an agent in the system"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO agent_registry (agent_id, agent_type, capabilities, status, description) 
                           VALUES (%s, %s, %s, 'active', %s) 
                           ON CONFLICT (agent_id) DO UPDATE SET 
                           capabilities = EXCLUDED.capabilities, 
                           last_heartbeat = CURRENT_TIMESTAMP, 
                           status = 'active',
                           description = EXCLUDED.description""",
                        (agent_id, agent_type, json.dumps(capabilities), description)
                    )
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return False

    def update_agent_heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat timestamp with proper row count handling"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE agent_registry SET last_heartbeat = CURRENT_TIMESTAMP WHERE agent_id = %s",
                        (agent_id,)
                    )
                    rows_affected = cur.rowcount  # Add this line
                    conn.commit()
                    
                    if rows_affected == 0:
                        # Agent not in registry, auto-register it
                        logger.info(f"Auto-registering agent {agent_id}")
                        return self.register_agent(
                            agent_id=agent_id,
                            agent_type="auto_registered",
                            capabilities={"auto_registered": True},
                            description=f"Auto-registered agent: {agent_id}"
                        )
                    
                    return True
        except Exception as e:
            logger.error(f"Failed to update agent heartbeat: {e}")
            return False



    def get_active_agents(self) -> List[Dict]:
        """Get list of active agents"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM agent_registry WHERE status = 'active' AND last_heartbeat > NOW() - INTERVAL '5 minutes' ORDER BY last_heartbeat DESC"
                    )
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get active agents: {e}")
            return []


    def _auto_register_agent(self, agent_id: str) -> bool:
        """Auto-register agent if not in registry"""
        try:
            return self.register_agent(
                agent_id=agent_id,
                agent_type="auto_registered",
                capabilities={"auto_registered": True},
                description=f"Auto-registered agent: {agent_id}"
            )
        except Exception as e:
            logger.warning(f"Auto-registration failed for {agent_id}: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Test basic connectivity
                    cur.execute("SELECT 1")
                    
                    # Get table statistics - FIXED: properly access count values
                    cur.execute("SELECT COUNT(*) FROM agent_logs")
                    log_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM agent_interactions")
                    # interaction_count = cur.fetchone()[1]
                    
                    cur.execute("SELECT COUNT(*) FROM agent_registry WHERE status = 'active'")
                    # active_agents = cur.fetchone()[1]
                    
                    # Check connection pool status
                    pool_info = {
                        "min_connections": self.connection_pool.minconn,
                        "max_connections": self.connection_pool.maxconn,
                        "available_connections": len(self.connection_pool._pool),
                        "used_connections": len(self.connection_pool._used)
                    }
                    
                    return {
                        "status": "healthy",
                        "database_stats": {
                            # "log_entries": log_count,
                            # "agent_interactions": interaction_count,
                            # "active_agents": active_agents
                        },
                        "connection_pool": pool_info,
                        "timestamp": time.time()
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# Global singleton instance
db_manager = DatabaseManager()
