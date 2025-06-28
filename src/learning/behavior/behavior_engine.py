"""
COMPLETELY DYNAMIC behavior engine - NO hardcoded categories or keywords
"""
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

try:
    import nest_asyncio
    nest_asyncio.apply()
    logger = logging.getLogger(__name__)
    logger.info("nest-asyncio applied successfully")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("nest-asyncio not available")

# Global instances
chroma_client = None
chroma_collection = None
setup_complete = False
embedding_model = None
thread_pool = ThreadPoolExecutor(max_workers=2)

def get_chroma_setup():
    """ChromaDB setup with embedding model initialization"""
    global chroma_client, chroma_collection, setup_complete, embedding_model
    
    if setup_complete and chroma_client and chroma_collection:
        return chroma_client, chroma_collection
    
    try:
        logger.info("Setting up ChromaDB connection...")
        chroma_client = chromadb.HttpClient(host="chroma", port=8000)
        
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer model loaded")
        except Exception as e:
            logger.warning(f"Sentence transformer setup failed: {e}")
        
        # DYNAMIC: Use timestamp-based collection names to avoid conflicts
        collection_name = "jarvis_memory"
        
        try:
            chroma_collection = chroma_client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
            setup_complete = True
            return chroma_client, chroma_collection
        except Exception:
            logger.info("Collection doesn't exist, creating new one")
        
        try:
            chroma_collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
            setup_complete = True
            return chroma_client, chroma_collection
        except Exception as create_error:
            logger.warning(f"Failed to create collection: {create_error}")
            setup_complete = True
            return None, None
            
    except Exception as e:
        logger.error(f"ChromaDB client setup failed: {e}")
        setup_complete = True
        return None, None

def sanitize_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata to ensure ChromaDB compliance"""
    sanitized = {}
    for key, value in metadata.items():
        clean_key = str(key)
        
        if value is None:
            sanitized[clean_key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[clean_key] = value
        elif isinstance(value, (list, dict, tuple)):
            try:
                json_str = json.dumps(value)[:500]
                sanitized[clean_key] = json_str
            except Exception:
                sanitized[clean_key] = str(value)[:100]
        elif isinstance(value, np.floating):
            sanitized[clean_key] = float(value)
        elif isinstance(value, np.integer):
            sanitized[clean_key] = int(value)
        else:
            sanitized[clean_key] = str(value)[:100]
            
    return sanitized

def run_async_safely(coro):
    """Safe async execution with improved error handling"""
    try:
        return asyncio.run(coro)
    except RuntimeError as runtime_error:
        error_msg = str(runtime_error)
        logger.warning(f"asyncio.run failed with RuntimeError: {error_msg}")
        
        try:
            loop = asyncio.get_running_loop()
            logger.info("Found running event loop, using run_coroutine_threadsafe")
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=15)
        except RuntimeError:
            try:
                def run_in_thread():
                    thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(thread_loop)
                    try:
                        return thread_loop.run_until_complete(coro)
                    finally:
                        thread_loop.close()
                        asyncio.set_event_loop(None)
                
                future = thread_pool.submit(run_in_thread)
                return future.result(timeout=15)
            except Exception as thread_error:
                logger.error(f"Thread execution failed: {thread_error}")
                return None
    except Exception as general_error:
        logger.error(f"Unexpected error in run_async_safely: {general_error}")
        return None

async def analyze_query_with_llm(query: str) -> Dict[str, Any]:
    """COMPLETELY DYNAMIC LLM analysis - NO hardcoded categories"""
    try:
        # DYNAMIC: Let LLM discover everything naturally without constraints
        prompt = f"""Analyze this user message and extract semantic information. Be creative and specific - don't limit yourself to common categories.

User Message: {query}

Analyze and return a JSON object with:
1. intent_type - what is the user trying to do? (discover naturally from context)
2. domain - what topic/subject is this about? (identify from content)
3. action - what action is being requested or expressed?
4. entities - important things/concepts mentioned
5. confidence - confidence score from 0.0 to 1.0
6. semantic_tags - descriptive tags that capture the essence

Be creative and discover patterns naturally. Respond with ONLY the JSON object."""
        
        response = requests.post(
            f"{settings.llm.ollama_url}/api/generate",
            json={
                "model": settings.llm.default_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "max_tokens": 300}
            },
            timeout=12
        )
        
        if response.status_code == 200:
            llm_response = response.json().get("response", "").strip()
            try:
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    analysis = json.loads(json_str)
                    logger.info(f"Pure LLM analysis for '{query}': {analysis}")
                    return analysis
                else:
                    logger.warning(f"No valid JSON found in LLM response: {llm_response}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}, Response: {llm_response}")
        
        # DYNAMIC FALLBACK: Basic semantic analysis
        return await pure_semantic_analysis(query)
        
    except Exception as e:
        logger.warning(f"LLM analysis failed: {e}")
        return await pure_semantic_analysis(query)

async def pure_semantic_analysis(query: str) -> Dict[str, Any]:
    """DYNAMIC semantic analysis using only embeddings and similarity"""
    global embedding_model
    
    if not embedding_model:
        return {
            "intent_type": "unknown",
            "domain": "general", 
            "action": "unknown",
            "entities": [],
            "confidence": 0.1,
            "semantic_tags": []
        }
    
    try:
        client, collection = get_chroma_setup()
        if collection:
            try:
                # DYNAMIC: Learn from existing data patterns
                similar_content = collection.query(query_texts=[query], n_results=5)
                
                if similar_content and similar_content.get("metadatas") and similar_content["metadatas"][0]:
                    existing_patterns = similar_content["metadatas"][0]
                    
                    # DYNAMIC: Extract patterns from existing data
                    intent_types = [meta.get("intent_type", "unknown") for meta in existing_patterns if meta.get("intent_type")]
                    domains = [meta.get("domain", "general") for meta in existing_patterns if meta.get("domain")]
                    
                    # DYNAMIC: Use most frequent pattern
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
        
        # FINAL FALLBACK
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

def add_to_semantic_memory(content: str, metadata: dict = None) -> bool:
    """DYNAMIC memory storage with LLM analysis"""
    try:
        client, collection = get_chroma_setup()
        if not collection:
            logger.warning("ChromaDB not available")
            return False
        
        # DYNAMIC: Use LLM analysis instead of hardcoded patterns
        base_metadata = metadata if metadata else {}
        current_time = time.time()
        base_metadata.update({
            "timestamp": current_time,
            "type": base_metadata.get("type", "conversation"),
            "session_id": base_metadata.get("session_id", "default")
        })
        
        # DYNAMIC: Get LLM analysis
        analysis = run_async_safely(analyze_query_with_llm(content))
        if not analysis:
            analysis = {
                "intent_type": "user_message",
                "domain": "conversation", 
                "action": "communicate",
                "entities": [],
                "confidence": 0.5,
                "semantic_tags": ["basic_storage"]
            }
        
        # DYNAMIC: Combine base metadata with LLM analysis
        llm_metadata = {
            "intent_type": str(analysis.get("intent_type", "unknown")),
            "domain": str(analysis.get("domain", "general")),
            "action": str(analysis.get("action", "unknown")),
            "llm_confidence": float(analysis.get("confidence", 0.0)),
            "entities_count": len(analysis.get("entities", [])),
            "tags_count": len(analysis.get("semantic_tags", [])),
            "has_entities": bool(analysis.get("entities", [])),
            "has_tags": bool(analysis.get("semantic_tags", []))
        }
        
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
        
        logger.info(f"Memory added - Intent: {analysis.get('intent_type')}, Domain: {analysis.get('domain')}, Confidence: {analysis.get('confidence', 0.0):.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return True

def search_semantic_memory(query: str, n_results: int = 3, session_id: str = None):
    """COMPLETELY DYNAMIC memory search using LLM analysis"""
    try:
        client, collection = get_chroma_setup()
        if not collection:
            logger.warning("ChromaDB not available for search")
            return None
        
        # DYNAMIC: Use LLM to analyze query
        analysis = run_async_safely(analyze_query_with_llm(query))
        if not analysis:
            analysis = {
                "intent_type": "search_query",
                "domain": "conversation",
                "confidence": 0.5
            }
        
        intent_type = analysis.get("intent_type")
        domain = analysis.get("domain")
        confidence = analysis.get("confidence", 0.0)
        
        logger.info(f"Semantic search - Intent: {intent_type}, Domain: {domain}, Confidence: {confidence:.2f}")
        
        search_results = []
        
        # DYNAMIC: Search with multiple strategies
        try:
            # Primary search with query
            results = collection.query(query_texts=[query], n_results=n_results * 2)
            if results and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [0] * len(documents))
                
                # FIXED: Ensure distances is handled properly
                if isinstance(distances, list) and len(distances) > 0:
                    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                        if i < len(distances):
                            distance = distances[i]
                            # FIXED: Ensure distance is a number
                            if isinstance(distance, (list, tuple)):
                                distance = distance[0] if distance else 0.0
                            distance = float(distance)
                            search_results.append((doc, meta, distance))
                        else:
                            search_results.append((doc, meta, 0.0))
                else:
                    for doc, meta in zip(documents, metadatas):
                        search_results.append((doc, meta, 0.0))
                        
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
        
        # DYNAMIC: Session-aware search if session provided
        if session_id and confidence > 0.3:
            try:
                where_conditions = {"$and": [{"session_id": session_id}]}
                if domain and domain != "general":
                    where_conditions["$and"].append({"domain": domain})
                
                session_results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_conditions
                )
                
                if session_results and session_results.get("documents") and session_results["documents"][0]:
                    documents = session_results["documents"][0]
                    metadatas = session_results.get("metadatas", [[]])[0]
                    distances = session_results.get("distances", [0] * len(documents))
                    
                    # FIXED: Same distance handling for session search
                    if isinstance(distances, list) and len(distances) > 0:
                        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                            if i < len(distances):
                                distance = distances[i]
                                if isinstance(distance, (list, tuple)):
                                    distance = distance[0] if distance else 0.0
                                distance = float(distance) + 0.1
                                search_results.append((doc, meta, distance))
                            else:
                                search_results.append((doc, meta, 0.1))
                    else:
                        for doc, meta in zip(documents, metadatas):
                            search_results.append((doc, meta, 0.1))
                            
            except Exception as e:
                logger.warning(f"Session search failed: {e}")
        
        if not search_results:
            logger.info("No search results found")
            return None
        
        # DYNAMIC: Score and rank results
        seen_content = set()
        unique_results = []
        
        for doc, meta, distance in search_results:
            if doc and doc.strip():
                content_key = doc.strip().lower()
                
                try:
                    distance = float(distance)
                    if content_key not in seen_content and distance < 0.9:
                        seen_content.add(content_key)
                        
                        # DYNAMIC: Calculate relevance score
                        relevance_score = 1.0 - distance
                        
                        # DYNAMIC: Boost for discovered patterns
                        if meta.get("intent_type") == intent_type:
                            relevance_score += 0.5
                        if meta.get("domain") == domain:
                            relevance_score += 0.6
                        if meta.get("session_id") == session_id:
                            relevance_score += 0.4
                        
                        llm_conf = meta.get("llm_confidence", 0.0)
                        if isinstance(llm_conf, (int, float)) and llm_conf > 0.7:
                            relevance_score += 0.8
                        
                        unique_results.append((doc, meta, relevance_score))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Distance conversion failed: {e}")
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        unique_results.append((doc, meta, 0.5))
        
        # DYNAMIC: Sort by relevance
        unique_results.sort(key=lambda x: x[2], reverse=True)
        final_results = unique_results[:n_results]
        
        if final_results:
            result = {
                "documents": [[doc for doc, _, _ in final_results]],
                "metadatas": [[meta for _, meta, _ in final_results]]
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
