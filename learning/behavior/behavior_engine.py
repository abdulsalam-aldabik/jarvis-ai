from sktime.forecasting.naive import NaiveForecaster
from sktime.datasets import load_airline
import wittgenstein as lw  # ✅ Use wittgenstein instead of ripperk
import pandas as pd
import uuid
import chromadb
import json
import time
import logging
from typing import Dict, Any, List, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

def train_time_series_forecaster(y):
    # Example: fit a naive forecaster
    forecaster = NaiveForecaster(strategy="mean")
    forecaster.fit(y)
    return forecaster

def learn_rules(X, y):
    # Example: fit RIPPER-k rule learner using wittgenstein
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        if hasattr(X, 'columns'):
            X = pd.DataFrame(X)
        else:
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Add target column to create the format wittgenstein expects
    data = X.copy()
    data['target'] = y
    
    # Create RIPPER classifier (k=2 for RIPPERk)
    clf = lw.RIPPER(k=2, prune_size=0.33, dl_allowance=64)
    clf.fit(data, class_feat='target')
    return clf

def propose_routine(candidate, user_approval_callback):
    # candidate: dict describing the routine
    # user_approval_callback: function to call for approval
    approved = user_approval_callback(candidate)
    if approved:
        print("Routine approved:", candidate)
    else:
        print("Routine rejected:", candidate)

# --- Enhanced ChromaDB Integration for Semantic Memory ---
def get_chroma_client():
    """Get ChromaDB client connection with error handling."""
    try:
        client = chromadb.HttpClient(host='chroma', port=8000)
        # Test connection
        client.heartbeat()
        return client
    except Exception as e:
        logger.error(f"ChromaDB connection failed: {e}")
        return None

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata to ensure ChromaDB compatibility.
    
    ChromaDB only accepts str, int, float, bool, or None values.
    """
    if not metadata:
        return {"source": "jarvis", "timestamp": str(time.time())}
    
    sanitized = {}
    
    for key, value in metadata.items():
        # Ensure key is a string
        clean_key = str(key)
        
        # Handle different value types
        if value is None:
            sanitized[clean_key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[clean_key] = value
        elif isinstance(value, (list, dict, tuple)):
            # Convert complex types to JSON strings
            sanitized[clean_key] = json.dumps(value)
        else:
            # Convert other types to strings
            sanitized[clean_key] = str(value)
    
    # Ensure we always have at least a timestamp
    if "timestamp" not in sanitized:
        sanitized["timestamp"] = str(time.time())
    
    # Ensure we have a source identifier
    if "source" not in sanitized:
        sanitized["source"] = "jarvis"
    
    return sanitized

def add_to_semantic_memory(content: str, metadata: dict = None, collection_name: str = "jarvis_memory"):
    """Add content to semantic memory using ChromaDB v2 API with proper metadata handling."""
    try:
        client = get_chroma_client()
        if not client:
            logger.warning("ChromaDB not available, skipping memory storage")
            return False
            
        collection = client.get_or_create_collection(collection_name)
        doc_id = str(uuid.uuid4())
        
        # Sanitize metadata for ChromaDB compatibility
        clean_metadata = sanitize_metadata(metadata)
        
        # Ensure content is not empty
        if not content or len(content.strip()) == 0:
            content = "Empty content"
        
        collection.add(
            documents=[content],
            metadatas=[clean_metadata],
            ids=[doc_id]
        )
        
        logger.info(f"Added to semantic memory: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add to semantic memory: {e}")
        return False

def search_semantic_memory(query: str, n_results: int = 5, collection_name: str = "jarvis_memory"):
    """Search semantic memory using ChromaDB v2 API."""
    try:
        client = get_chroma_client()
        if not client:
            logger.warning("ChromaDB not available for search")
            return []
            
        # Check if collection exists
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            logger.info(f"Collection {collection_name} doesn't exist yet")
            return []
        
        # Ensure query is not empty
        if not query or len(query.strip()) == 0:
            query = "general query"
            
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        logger.info(f"Semantic memory search: {len(results['documents'][0]) if results['documents'] else 0} results")
        return results
        
    except Exception as e:
        logger.error(f"Semantic memory search failed: {e}")
        return []

# Example usage:
if __name__ == "__main__":
    # Time series forecasting
    y = load_airline()
    model = train_time_series_forecaster(y)
    print("Time series forecast:", model.predict([len(y), len(y)+1, len(y)+2]))

    # For RIPPER-k, create example tabular data
    import numpy as np
    
    # Example dataset for rule learning
    np.random.seed(42)
    X_example = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.normal(60, 15, 100),
        'pressure': np.random.normal(1013, 10, 100)
    })
    
    # Create target based on some rules (for demonstration)
    y_example = ((X_example['temperature'] > 25) & 
                 (X_example['humidity'] < 50)).astype(int)
    
    # Learn rules
    rules = learn_rules(X_example, y_example)
    print("Learned rules:")
    print(rules.ruleset_)
    
    # Example of routine proposal
    def mock_user_approval(candidate):
        print(f"Do you approve this routine? {candidate}")
        return True  # Mock approval
    
    sample_routine = {
        "name": "Morning Workout",
        "time": "07:00",
        "duration": "30 minutes",
        "activities": ["stretching", "cardio"]
    }
    
    propose_routine(sample_routine, mock_user_approval)
    
    # Test semantic memory with proper metadata
    print("\nTesting semantic memory...")
    test_metadata = {
        "type": "test",
        "user_id": "test_user",
        "complexity_level": 1,
        "active": True
    }
    
    if add_to_semantic_memory("Test memory content", test_metadata):
        results = search_semantic_memory("test")
        print(f"Search results: {len(results.get('documents', [[]])[0])} documents found")
