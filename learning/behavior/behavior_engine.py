import chromadb
import uuid
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from config.settings import settings
import numpy as np
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# ✅ SOLUTION 1: Add nest-asyncio support (from search results)
try:
    import nest_asyncio
    nest_asyncio.apply()
    logger = logging.getLogger(__name__)
    logger.info("nest-asyncio applied successfully")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("nest-asyncio not available - install with: pip install nest-asyncio")

# Global ChromaDB instances
_chroma_client = None
_chroma_collection = None
_setup_complete = False
_embedding_model = None
_thread_pool = ThreadPoolExecutor(max_workers=2)

def get_chroma_setup():
    """ChromaDB setup with embedding model initialization"""
    global _chroma_client, _chroma_collection, _setup_complete, _embedding_model
    
    if _setup_complete and _chroma_client and _chroma_collection:
        return _chroma_client, _chroma_collection
    
    if _setup_complete:
        return _chroma_client, _chroma_collection
    
    try:
        logger.info("Setting up ChromaDB connection...")
        
        _chroma_client = chromadb.HttpClient(host='chroma', port=8000)
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded")
        except Exception as e:
            logger.warning(f"Sentence transformer setup failed: {e}")
        
        # Setup embedding function for ChromaDB
        embedding_func = None
        try:
            from chromadb.utils import embedding_functions
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.warning(f"SentenceTransformer setup failed: {e}")
        
        collection_name = "jarvis_memory"
        
        # Try to get existing collection
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
            
        except Exception:
            logger.info("Collection doesn't exist, creating new one")
        
        # Create new collection
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
            
            # Reset and recreate
            try:
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
        
        # Fallback
        try:
            _chroma_collection = _chroma_client.get_collection(name=collection_name)
            logger.warning("Using existing collection without custom embedding")
            _setup_complete = True
            return _chroma_client, _chroma_collection
        except Exception:
            logger.error("All collection setup methods failed")
        
        _setup_complete = True
        return None, None
        
    except Exception as e:
        logger.error(f"ChromaDB client setup failed: {e}")
        _setup_complete = True
        return None, None

def sanitize_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata to ensure ChromaDB compliance
    Based on search results [4]: values can only be strings, integers, floats, or booleans
    """
    sanitized = {}
    
    for key, value in metadata.items():
        # Ensure key is string
        clean_key = str(key)
        
        # ✅ CRITICAL: Ensure value types are ChromaDB-compliant
        if value is None:
            sanitized[clean_key] = ""  # Convert None to empty string
        elif isinstance(value, (str, int, float, bool)):
            # ✅ VALID: Direct ChromaDB-compliant types
            sanitized[clean_key] = value
        elif isinstance(value, (list, dict, tuple)):
            # ✅ FIXED: Convert complex types to JSON strings (max 500 chars)
            try:
                json_str = json.dumps(value)[:500]  # Limit length
                sanitized[clean_key] = json_str
            except Exception:
                sanitized[clean_key] = str(value)[:100]
        elif isinstance(value, np.floating):
            # ✅ FIXED: Handle numpy types (from search results [2])
            sanitized[clean_key] = float(value)
        elif isinstance(value, np.integer):
            sanitized[clean_key] = int(value)
        else:
            # ✅ FALLBACK: Convert to string with length limit
            sanitized[clean_key] = str(value)[:100]
    
    return sanitized

def run_async_safely(coro):
    """
    Run async function safely with IMPROVED ERROR HANDLING
    """
    try:
        # ✅ SOLUTION 1: Try direct asyncio.run first (works with nest-asyncio)
        return asyncio.run(coro)
        
    except RuntimeError as runtime_error:
        # ✅ DETAILED ERROR LOGGING
        error_msg = str(runtime_error)
        logger.warning(f"asyncio.run failed with RuntimeError: {error_msg}")
        
        # ✅ SOLUTION 2: Check if there's a running loop and use run_coroutine_threadsafe
        try:
            loop = asyncio.get_running_loop()
            logger.info("Found running event loop, using run_coroutine_threadsafe")
            
            # Use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=15)
            
        except RuntimeError as loop_error:
            logger.warning(f"No running loop found: {loop_error}")
            
            # ✅ SOLUTION 3: Create new event loop
            try:
                logger.info("Creating new event loop in thread")
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(coro)
                    return result
                finally:
                    new_loop.close()
                    
            except Exception as new_loop_error:
                logger.error(f"New loop creation failed: {new_loop_error}")
                
                # ✅ SOLUTION 4: ThreadPoolExecutor fallback
                def run_in_thread():
                    thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(thread_loop)
                    try:
                        return thread_loop.run_until_complete(coro)
                    finally:
                        thread_loop.close()
                        asyncio.set_event_loop(None)
                
                try:
                    future = _thread_pool.submit(run_in_thread)
                    return future.result(timeout=15)
                except Exception as thread_error:
                    logger.error(f"Thread execution failed: {thread_error}")
                    return None
        
        except Exception as threadsafe_error:
            logger.error(f"run_coroutine_threadsafe failed: {threadsafe_error}")
            return None
    
    except Exception as general_error:
        logger.error(f"Unexpected error in run_async_safely: {general_error}")
        return None

async def analyze_query_with_llm(query: str) -> Dict[str, Any]:
    """
    Pure LLM-based analysis with NO hardcoded categories or keywords
    """
    try:
        # ✅ PURE LLM: Let the LLM determine everything dynamically
        prompt = f"""Analyze this user query and extract semantic information. Do not use predefined categories - discover the intent and domain naturally from the content.

Query: "{query}"

Analyze and return a JSON object with:
1. "intent_type" - what is the user trying to do? Discover this from context, don't use predefined categories
2. "domain" - what topic is this about? Identify from the content itself
3. "action" - what action is being requested or stated?
4. "entities" - important entities mentioned
5. "confidence" - confidence score from 0.0 to 1.0
6. "semantic_tags" - list of semantic tags that describe the query

Be creative and specific - don't limit yourself to common categories. Respond with ONLY the JSON object.
"""

        response = requests.post(
            f"{settings.llm.ollama_url}/api/generate",
            json={
                "model": settings.llm.default_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 300
                }
            },
            timeout=12
        )

        if response.status_code == 200:
            llm_response = response.json().get("response", "").strip()
            
            try:
                # Clean up response (remove any non-JSON text)
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    analysis = json.loads(json_str)
                    logger.info(f"Pure LLM analysis for '{query}': {analysis}")
                    return analysis
                else:
                    logger.warning(f"No valid JSON found in LLM response: {llm_response}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}, Response: {llm_response}")
        
    except Exception as e:
        logger.warning(f"LLM analysis failed: {e}")
    
    # ✅ PURE SEMANTIC FALLBACK: Use only embeddings, no keywords
    return await _pure_semantic_analysis(query)

async def _pure_semantic_analysis(query: str) -> Dict[str, Any]:
    """
    Completely semantic analysis using only embeddings and similarity
    NO hardcoded keywords or categories
    """
    global _embedding_model
    
    if not _embedding_model:
        # Ultimate fallback - minimal structure
        return {
            "intent_type": "unknown",
            "domain": "general",
            "action": "unknown",
            "entities": [],
            "confidence": 0.1,
            "semantic_tags": []
        }
    
    try:
        # ✅ PURE SEMANTIC: Learn from existing data patterns
        client, collection = get_chroma_setup()
        
        if collection:
            # Query similar content to learn patterns
            try:
                similar_content = collection.query(
                    query_texts=[query],
                    n_results=5
                )
                
                if similar_content and similar_content.get('metadatas') and similar_content['metadatas'][0]:
                    # Learn from existing patterns
                    existing_patterns = similar_content['metadatas'][0]
                    
                    # Extract the most common patterns
                    intent_types = [meta.get('intent_type', 'unknown') for meta in existing_patterns if meta.get('intent_type')]
                    domains = [meta.get('domain', 'general') for meta in existing_patterns if meta.get('domain')]
                    
                    # Use the most frequent pattern with some confidence
                    most_common_intent = max(set(intent_types), key=intent_types.count) if intent_types else "unknown"
                    most_common_domain = max(set(domains), key=domains.count) if domains else "general"
                    
                    return {
                        "intent_type": most_common_intent,
                        "domain": most_common_domain,
                        "action": "inferred_from_similar",
                        "entities": [],
                        "confidence": 0.6,
                        "semantic_tags": ["learned_pattern"]
                    }
            except Exception as e:
                logger.warning(f"Pattern learning failed: {e}")
        
        # Final fallback
        return {
            "intent_type": "communication",
            "domain": "general",
            "action": "express",
            "entities": [],
            "confidence": 0.3,
            "semantic_tags": ["semantic_fallback"]
        }
        
    except Exception as e:
        logger.warning(f"Pure semantic analysis failed: {e}")
        return {
            "intent_type": "unknown",
            "domain": "general", 
            "action": "unknown",
            "entities": [],
            "confidence": 0.1,
            "semantic_tags": []
        }

async def generate_dynamic_expansions(query: str, analysis: Dict[str, Any]) -> List[str]:
    """
    Generate expansions using pure LLM creativity - NO templates or patterns
    """
    expansions = [query]
    
    try:
        confidence = analysis.get("confidence", 0.0)
        
        if confidence < 0.3:
            return expansions
        
        # ✅ PURE LLM: Let LLM be creative with expansions
        expansion_prompt = f"""Generate 4 different ways to express or search for the same concept as this query. Be creative and think of how someone might phrase this differently or what related terms might be used.

Original: "{query}"
Intent: {analysis.get('intent_type', 'unknown')}
Domain: {analysis.get('domain', 'unknown')}

Generate variations that capture the same semantic meaning but use different words, structures, or perspectives. Think outside the box.

Return only the variations, one per line:
"""

        response = requests.post(
            f"{settings.llm.ollama_url}/api/generate",
            json={
                "model": settings.llm.default_model,
                "prompt": expansion_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Higher creativity
                    "max_tokens": 150
                }
            },
            timeout=10
        )

        if response.status_code == 200:
            llm_expansions = response.json().get("response", "").strip()
            
            for line in llm_expansions.split('\n'):
                expansion = line.strip()
                # Remove numbering, bullets, etc.
                expansion = expansion.lstrip('1234567890.-• ')
                
                if expansion and expansion != query and len(expansion) > 3:
                    expansions.append(expansion)
        
        logger.info(f"Generated {len(expansions)} creative expansions for '{query}'")
        return expansions[:6]  # Limit to 6 total
        
    except Exception as e:
        logger.warning(f"Creative expansion generation failed: {e}")
        return expansions

def add_to_semantic_memory(content: str, metadata: dict = None) -> bool:
    """Add to memory with pure LLM analysis - FIXED CHROMADB METADATA COMPLIANCE"""
    try:
        client, collection = get_chroma_setup()
        if not collection:
            logger.warning("ChromaDB not available")
            return False
        
        # Prepare base metadata
        base_metadata = {}
        if metadata:
            # ✅ CRITICAL: Sanitize input metadata first
            base_metadata = sanitize_metadata_for_chromadb(metadata)
        
        current_time = time.time()
        base_metadata.update({
            "timestamp": current_time,  # float - ChromaDB compliant
            "type": base_metadata.get("type", "conversation"),  # string
            "session_id": base_metadata.get("session_id", "default")  # string
        })
        
        # ✅ IMPROVED: Use safe async runner with better error handling
        analysis = run_async_safely(analyze_query_with_llm(content))
        
        if not analysis:
            # ✅ BETTER FALLBACK: Still store with basic analysis
            analysis = {
                "intent_type": "user_message",
                "domain": "conversation",
                "action": "communicate",
                "entities": [],
                "confidence": 0.5,
                "semantic_tags": ["basic_storage"]
            }
            logger.info("Using fallback analysis for memory storage")
        
        # ✅ CRITICAL FIX: Ensure all LLM analysis data is ChromaDB-compliant
        llm_metadata = {
            "intent_type": str(analysis.get("intent_type", "unknown")),  # string
            "domain": str(analysis.get("domain", "general")),  # string
            "action": str(analysis.get("action", "unknown")),  # string
            "llm_confidence": float(analysis.get("confidence", 0.0)),  # float
            # ✅ FIXED: Convert complex types to simple strings
            "entities_count": len(analysis.get("entities", [])),  # int
            "tags_count": len(analysis.get("semantic_tags", [])),  # int
            "has_entities": bool(analysis.get("entities", [])),  # boolean
            "has_tags": bool(analysis.get("semantic_tags", []))  # boolean
        }
        
        # ✅ CRITICAL: Sanitize the combined metadata
        clean_metadata = sanitize_metadata_for_chromadb({**base_metadata, **llm_metadata})
        
        # Store main content
        searchable_content = content.strip()
        base_id = str(int(current_time * 1000))
        session_short = clean_metadata.get("session_id", "default")[:8]
        doc_id = f"doc_{base_id}_{session_short}"
        
        collection.add(
            documents=[searchable_content],
            metadatas=[clean_metadata],
            ids=[doc_id]
        )
        
        # ✅ STORE VARIATIONS: Only if confidence is reasonable
        confidence = analysis.get("confidence", 0.0)
        if confidence > 0.4:
            try:
                variations = run_async_safely(generate_dynamic_expansions(content, analysis))
                
                if variations and len(variations) > 1:
                    for i, variation in enumerate(variations[1:], 1):
                        # ✅ CRITICAL: Ensure variation metadata is also compliant
                        variation_metadata = clean_metadata.copy()
                        variation_metadata["content_type"] = "creative_variation"  # string
                        variation_metadata["original_doc_id"] = doc_id  # string
                        variation_metadata["variation_index"] = i  # int
                        
                        # ✅ SANITIZE: Ensure compliance
                        variation_metadata = sanitize_metadata_for_chromadb(variation_metadata)
                        
                        collection.add(
                            documents=[variation],
                            metadatas=[variation_metadata],
                            ids=[f"{doc_id}_creative_{i}"]
                        )
            except Exception as e:
                logger.warning(f"Failed to generate creative variations: {e}")
        
        logger.info(f"Memory added - Intent: {analysis.get('intent_type')}, Domain: {analysis.get('domain')}, Confidence: {confidence:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return True

def search_semantic_memory(query: str, n_results: int = 3, session_id: str = None):
    """Search memory using pure semantic understanding - IMPROVED ASYNCIO HANDLING"""
    try:
        client, collection = get_chroma_setup()
        if not collection:
            logger.warning("ChromaDB not available for search")
            return None
        
        # ✅ IMPROVED: Use safe async runner for LLM analysis
        analysis = run_async_safely(analyze_query_with_llm(query))
        
        if not analysis:
            # ✅ BETTER FALLBACK: Use basic semantic search
            analysis = {
                "intent_type": "search_query",
                "domain": "conversation",
                "confidence": 0.5
            }
            logger.info("Using fallback analysis for memory search")
        
        expansions = run_async_safely(generate_dynamic_expansions(query, analysis))
        
        if not expansions:
            expansions = [query]
        
        intent_type = analysis.get("intent_type", "")
        domain = analysis.get("domain", "")
        confidence = analysis.get("confidence", 0.0)
        
        logger.info(f"Semantic search - Intent: {intent_type}, Domain: {domain}, Confidence: {confidence:.2f}")
        
        search_results = []
        
        # Search with each creative expansion
        for expansion in expansions:
            try:
                results = collection.query(
                    query_texts=[expansion],
                    n_results=n_results * 2
                )
                
                if results and results.get('documents') and results['documents'][0]:
                    search_results.extend(list(zip(
                        results['documents'][0],
                        results.get('metadatas', [[]])[0],
                        results.get('distances', [[]])[0] if results.get('distances') else [0] * len(results['documents'][0])
                    )))
            except Exception as e:
                logger.warning(f"Search failed for expansion '{expansion}': {e}")
        
        # Session-aware search with discovered domain
        if session_id and confidence > 0.3:
            try:
                recent_time = time.time() - 3600
                where_conditions = {"$and": [{"session_id": session_id}, {"timestamp": {"$gte": recent_time}}]}
                
                # Use discovered domain dynamically
                if domain and domain != "general" and domain != "unknown":
                    where_conditions["$and"].append({"domain": domain})
                
                session_results = collection.query(
                    query_texts=expansions,
                    n_results=n_results,
                    where=where_conditions
                )
                
                if session_results and session_results.get('documents') and session_results['documents'][0]:
                    session_items = list(zip(
                        session_results['documents'][0],
                        session_results.get('metadatas', [[]])[0],
                        [d * 0.2 for d in session_results.get('distances', [[]])[0]] if session_results.get('distances') else [0] * len(session_results['documents'][0])
                    ))
                    search_results = session_items + search_results
            except Exception as e:
                logger.warning(f"Session search failed: {e}")
        
        if not search_results:
            logger.info("No search results found")
            return None
        
        # ✅ PURE SEMANTIC: Rank by discovered patterns
        seen_content = set()
        unique_results = []
        
        for doc, meta, distance in search_results:
            if doc and doc.strip():
                content_key = doc.strip().lower()
                if content_key not in seen_content and distance < 0.9:
                    seen_content.add(content_key)
                    
                    # Dynamic scoring based on discovered attributes
                    relevance_score = distance
                    
                    # Boost for discovered intent match
                    if meta.get("intent_type") == intent_type:
                        relevance_score *= 0.5
                    
                    # Boost for discovered domain match
                    if meta.get("domain") == domain:
                        relevance_score *= 0.6
                    
                    # Boost for session match
                    if meta.get("session_id") == session_id:
                        relevance_score *= 0.4
                    
                    # Boost for high LLM confidence
                    llm_conf = meta.get("llm_confidence", 0.0)
                    if isinstance(llm_conf, (int, float)) and llm_conf > 0.7:
                        relevance_score *= 0.8
                    
                    unique_results.append((doc, meta, relevance_score))
        
        # Sort by pure semantic relevance
        unique_results.sort(key=lambda x: x[2])
        final_results = unique_results[:n_results]
        
        if final_results:
            result = {
                'documents': [[doc for doc, _, _ in final_results]],
                'metadatas': [[meta for _, meta, _ in final_results]]
            }
            
            logger.info(f"Semantic search returned {len(final_results)} results")
            
            for i, (doc, meta, score) in enumerate(final_results):
                logger.info(f"Result {i}: {doc[:50]}... (score: {score:.3f})")
            
            return result
        
        logger.info("No relevant results found")
        return None
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return None

# Keep other functions unchanged...
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
    """Reset memory system"""
    global _chroma_client, _chroma_collection, _setup_complete, _embedding_model
    _chroma_client = None
    _chroma_collection = None
    _setup_complete = False
    _embedding_model = None
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
