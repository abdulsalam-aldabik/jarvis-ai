"""
Structured logging configuration for AutoGen + LangGraph hybrid system
"""
import json
import logging
import time
from typing import Dict, Any, Optional

def log_structured(event_type: str, **kwargs):
    """Log structured events with consistent format"""
    logger = logging.getLogger("jarvis")
    
    log_entry = {
        "event_type": event_type,
        "timestamp": time.time(),
        **kwargs
    }
    
    logger.info(json.dumps(log_entry, default=str))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
