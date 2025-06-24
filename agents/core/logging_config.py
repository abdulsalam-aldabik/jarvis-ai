import logging
import json

logger = logging.getLogger("jarvis")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_structured(event, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))
