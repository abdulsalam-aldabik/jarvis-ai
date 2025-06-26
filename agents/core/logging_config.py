import logging
import json

def setup_logger():
    logger = logging.getLogger("jarvis")
    
    # ✅ FIXED: Only add handler if none exist
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent root logger duplication
    
    return logger

logger = setup_logger()

def log_structured(event, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))
