import chromadb
import uuid
import json
import time
import logging
from typing import Dict, Any, List, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

# Global ChromaDB instances - single point of truth
_chroma_client = None
_chroma_collection = None
_setup_complete = False

def get_chroma_setup():
    """Get ChromaDB with manual collection management"""
    global _chroma_client, _chroma_collection, _setup_complete
    
    # Return cached instances if setup is complete
    if _setup_complete and _chroma_client and _chroma_collection:
        return _chroma_client, _chroma_collection
    
    # Prevent re-entry during setup
    if _setup_complete:
        return _chroma_client, _chroma_collection
    
    try:
        logger.info("Setting up ChromaDB connection...")
        
        # Create client
        _chroma_client = chromadb.HttpClient(host='chroma', port=8000)
        
        # Set up embedding function
        embedding_func = None
        try:
            from chromadb.utils import embedding_functions
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("SentenceTransformer embeddings configured")
        except Exception as e:
            logger.warning(f"SentenceTransformer setup failed: {e}")
        
        collection_name = "jarvis_memory"
        
        # Method 1: Try to get existing collection first
        try:
            if embedding_func:
                _chroma_collection = _chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_func
                )
            else:
                _chroma_collection = _chroma_client.get_collection(name=collection_name)
            
            logger.info(f"Using existing collection: {collection_name}")
            _setup_complete = True
            return _chroma_client, _chroma_collection
            
        except Exception as get_error:
            logger.info(f"Collection doesn't exist or incompatible: {get_error}")
        
        # Method 2: Create new collection if getting failed
        try:
            if embedding_func:
                _chroma_collection = _chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                _chroma_collection = _chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            logger.info(f"Created new collection: {collection_name}")
            _setup_complete = True
            return _chroma_client, _chroma_collection
            
        except Exception as create_error:
            logger.warning(f"Failed to create collection: {create_error}")
        
        # Method 3: Delete and recreate if there's a conflict
        try:
            logger.info("Attempting to reset collection...")
            _chroma_client.delete_collection(name=collection_name)
            
            if embedding_func:
                _chroma_collection = _chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                _chroma_collection = _chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            logger.info(f"Reset and created collection: {collection_name}")
            _setup_complete = True
            return _chroma_client, _chroma_collection
            
        except Exception as reset_error:
            logger.error(f"Failed to reset collection: {reset_error}")
        
        # Method 4: Fallback to existing collection without embedding function
        try:
            _chroma_collection = _chroma_client.get_collection(name=collection_name)
            logger.warning("Using existing collection without custom embedding")
            _setup_complete = True
            return _chroma_client, _chroma_collection
            
        except Exception as fallback_error:
            logger.error(f"All collection setup methods failed: {fallback_error}")
        
        # Complete failure
        _setup_complete = True  # Prevent retries
        return None, None
        
    except Exception as e:
        logger.error(f"ChromaDB client setup failed: {e}")
        _setup_complete = True  # Prevent retries
        return None, None

def add_to_semantic_memory(content: str, metadata: dict = None) -> bool:
    """Add to memory with deduplication"""
    try:
        client, collection = get_chroma_setup()
        if not collection:
            logger.warning("ChromaDB not available")
            return False
        
        # Prepare metadata
        clean_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)[:100]
        
        clean_metadata["timestamp"] = str(time.time())
        clean_metadata["type"] = clean_metadata.get("type", "conversation")
        
        # Simple deduplication by content hash
        content_key = content.strip().lower()
        content_hash = str(abs(hash(content_key)))[:10]
        clean_metadata["content_hash"] = content_hash
        
        # Check for recent duplicates
        try:
            recent_results = collection.query(
                query_texts=[content_key],
                n_results=1,
                where={"content_hash": content_hash}
            )
            
            if (recent_results and recent_results.get('documents') and 
                recent_results['documents'][0]):
                logger.info("Duplicate content detected, skipping")
                return True
                
        except Exception:
            pass  # Ignore duplicate check errors
        
        # Add to collection
        doc_id = f"doc_{int(time.time())}_{content_hash}"
        
        collection.add(
            documents=[content],
            metadatas=[clean_metadata],
            ids=[doc_id]
        )
        
        logger.info(f"Memory added: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return False

def search_semantic_memory(query: str, n_results: int = 3):
    """Search memory with proper error handling"""
    try:
        client, collection = get_chroma_setup()
        if not collection:
            logger.warning("ChromaDB not available for search")
            return None
        
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, 5)
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            logger.info("No search results found")
            return None
        
        documents = results['documents'][0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0] if results.get('distances') else None
        
        # Filter by relevance
        filtered_docs = []
        filtered_meta = []
        
        for i, doc in enumerate(documents):
            # Include if no distances or distance is reasonable
            if not distances or distances[i] < 0.9:
                filtered_docs.append(doc)
                filtered_meta.append(metadatas[i] if i < len(metadatas) else {})
        
        if filtered_docs:
            result = {
                'documents': [filtered_docs],
                'metadatas': [filtered_meta]
            }
            logger.info(f"Memory search: {len(filtered_docs)} relevant results")
            return result
        
        logger.info("No relevant results found")
        return None
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        return None

def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics"""
    try:
        client, collection = get_chroma_setup()
        if not collection:
            return {"status": "unavailable", "count": 0}
        
        count = collection.count()
        return {
            "status": "available",
            "count": count,
            "setup_complete": _setup_complete
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def reset_memory_system():
    """Reset memory system (for testing)"""
    global _chroma_client, _chroma_collection, _setup_complete
    _chroma_client = None
    _chroma_collection = None
    _setup_complete = False
    logger.info("Memory system reset")

# Keep ML functions for compatibility
def train_time_series_forecaster(y):
    from sktime.forecasting.naive import NaiveForecaster
    forecaster = NaiveForecaster(strategy="mean")
    forecaster.fit(y)
    return forecaster

def learn_rules(X, y):
    import wittgenstein as lw
    import pandas as pd
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    data = X.copy()
    data['target'] = y
    
    clf = lw.RIPPER(k=2, prune_size=0.33, dl_allowance=64)
    clf.fit(data, class_feat='target')
    return clf

def propose_routine(candidate, user_approval_callback):
    approved = user_approval_callback(candidate)
    if approved:
        print("Routine approved:", candidate)
    else:
        print("Routine rejected:", candidate)
