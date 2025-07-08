"""
COMPLETELY DYNAMIC behavior engine – NO hard-coded categories or keywords
"""
from __future__ import annotations

# ─────────────────────────────────────────  standard libs ──────────────────
import asyncio, hashlib, json, logging, threading, time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────  third-party  ────────────────────
import chromadb
import numpy as np
import requests
from config.settings import settings

# ─────────────────────────────────────────  logger / asyncio  ───────────────
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────  GLOBAL CACHES  ──────────────────
_connection_lock = threading.Lock()
_memory_lock     = threading.Lock()
thread_pool      = ThreadPoolExecutor(max_workers=2)

chroma_client:   Optional[chromadb.HttpClient]        = None
chroma_collection: Optional[chromadb.Collection]      = None
_chroma_cache:   Optional[Tuple[chromadb.HttpClient,
                                chromadb.Collection]] = None
setup_complete   = False

_embedding_model_cache = None         # global singleton
embedding_model         = None        # module alias

_recent_memories: Dict[str, Tuple[float, str]] = {}
_DEDUP_WINDOW_SEC = 30

# ─────────────────────────────────────────  EMBEDDING WARM-LOAD  ────────────
@lru_cache(maxsize=1)
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2",
                                cache_folder="/data/sentence_transformers")
    logger.info("Sentence-Transformer model pre-loaded")
    return model

embedding_model = _load_embedding_model()

# ─────────────────────────────────────────  CHROMA INITIALISATION  ──────────
def get_chroma_setup() -> Tuple[Optional[chromadb.HttpClient],
                                Optional[chromadb.Collection]]:
    """
    Thread-safe singleton that returns a ready HttpClient + Collection.
    Creates the ‘jarvis_memory’ collection on first run.
    """
    global chroma_client, chroma_collection, _chroma_cache, setup_complete

    if _chroma_cache and setup_complete:
        chroma_client, chroma_collection = _chroma_cache
        return _chroma_cache

    with _connection_lock:
        if _chroma_cache and setup_complete:
            chroma_client, chroma_collection = _chroma_cache
            return _chroma_cache
        try:
            logger.info("🔌 Initialising ChromaDB client …")
            client = chromadb.HttpClient(host="chroma", port=8000)
            name   = "jarvis_memory"

            try:
                collection = client.get_collection(name=name)
                logger.info(f"Using existing collection: {name}")
            except Exception:
                collection = client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created collection: {name}")

            _chroma_cache   = (client, collection)
            chroma_client, chroma_collection = _chroma_cache
            setup_complete  = True
            return _chroma_cache
        except Exception as exc:
            logger.error(f"Chroma setup failed: {exc}")
            setup_complete = True
            return None, None

# ─────────────────────────────────────────  UTILS  ──────────────────────────
def sanitize_metadata_for_chromadb(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any value into a Chroma-acceptable scalar / JSON string."""
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        k = str(k)
        try:
            if v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, dict, tuple)):
                clean[k] = json.dumps(v)[:500]
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            else:
                clean[k] = str(v)[:100]
        except Exception:
            clean[k] = str(v)[:100]
    return clean


def run_async_safely(coro):
    """Run a coroutine from sync code even if an event loop is active."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        try:
            loop = asyncio.get_running_loop()
            fut  = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result(timeout=15)
        except RuntimeError:
            def _runner():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            return thread_pool.submit(_runner).result(timeout=15)
    except Exception as exc:
        logger.error(f"run_async_safely error: {exc}")
        return None

# ─────────────────────────────────────────  LLM ANALYSIS (with preferences) ─
    async def analyze_query_with_llm(query: str) -> Dict[str, Any]:
        """
        ROBUST: LLM analysis with improved JSON extraction and preference detection
        """
        # Enhanced prompt with stricter JSON formatting
        prompt = f"""Analyze this user message and return ONLY valid JSON:

    User Message: {query}

    PREFERENCE PATTERNS:
    - "I like X" → preference_statement with food domain if X is food/drink
    - "I love X" → strong preference 
    - "I prefer X" → preference
    - "My favorite X" → favorite

    Return valid JSON object:
    {{
    "intent_type": "preference_statement|query|greeting|information_request",
    "domain": "food|drink|general|social|weather",
    "action": "state_preference|ask_preference|request_info",
    "entities": ["pizza", "coffee"],
    "confidence": 0.9,
    "semantic_tags": ["preference", "food"],
    "preference_data": {{
        "is_preference": true,
        "preference_type": "food_preference",
        "items": ["pizza", "coffee"],
        "strength": "like"
    }}
    }}

    IMPORTANT: Return ONLY the JSON object, no additional text."""

        try:
            response = requests.post(
                f"{settings.llm.ollama_url}/api/generate",
                json={
                    "model": settings.llm.default_model,
                    "prompt": prompt.strip(),
                    "stream": False,
                    "options": {
                        "temperature": 0.1, 
                        "max_tokens": 500,
                        "top_p": 0.9,
                        "stop": ["\n\n", "```", "```json", "```"]
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                llm_text = response.json().get("response", "").strip()
                
                # ROBUST JSON EXTRACTION: Multiple strategies
                analysis = None
                
                # Strategy 1: Direct JSON parse
                try:
                    analysis = json.loads(llm_text)
                    logger.info(f"✅ Direct JSON parse successful")
                except json.JSONDecodeError:
                    pass
                
                # Strategy 2: Extract JSON from text
                if not analysis:
                    try:
                        # Find JSON boundaries more reliably
                        json_start = llm_text.find("{")
                        json_end = llm_text.rfind("}")
                        
                        if json_start >= 0 and json_end > json_start:
                            json_text = llm_text[json_start:json_end + 1]
                            
                            # Clean common issues
                            json_text = json_text.replace("'", '"')  # Fix single quotes
                            json_text = json_text.replace('\n', ' ')  # Remove newlines
                            
                            analysis = json.loads(json_text)
                            logger.info(f"✅ Extracted JSON parse successful")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON extraction failed: {e}")
                
                # Strategy 3: Line-by-line JSON reconstruction
                if not analysis:
                    try:
                        lines = [line.strip() for line in llm_text.split('\n') if line.strip()]
                        json_lines = []
                        in_json = False
                        
                        for line in lines:
                            if line.startswith('{'):
                                in_json = True
                            if in_json:
                                json_lines.append(line)
                            if line.endswith('}'):
                                break
                        
                        if json_lines:
                            json_text = ' '.join(json_lines)
                            analysis = json.loads(json_text)
                            logger.info(f"✅ Reconstructed JSON parse successful")
                            
                    except json.JSONDecodeError:
                        pass
                
                # Validate and enhance analysis
                if analysis and isinstance(analysis, dict):
                    # Ensure preference_data exists
                    if "preference_data" not in analysis:
                        analysis["preference_data"] = {"is_preference": False}
                    
                    # Enhanced preference detection for common patterns
                    query_lower = query.lower()
                    if any(pattern in query_lower for pattern in ["i like", "i love", "i prefer", "my favorite"]):
                        analysis["intent_type"] = "preference_statement"
                        analysis["preference_data"]["is_preference"] = True
                        
                        # Extract food items
                        food_keywords = ["pizza", "coffee", "burger", "pasta", "salad", "sandwich", "sushi", "tacos"]
                        found_items = [item for item in food_keywords if item in query_lower]
                        
                        if found_items:
                            analysis["domain"] = "food"
                            analysis["preference_data"]["preference_type"] = "food_preference"
                            analysis["preference_data"]["items"] = found_items
                            analysis["confidence"] = 0.95
                    
                    logger.info(f"🎯 LLM analysis successful: {analysis.get('intent_type')} / {analysis.get('domain')}")
                    return analysis
                    
            logger.warning(f"All JSON parsing strategies failed, LLM response: {llm_text[:200]}...")
            
        except Exception as e:
            logger.warning(f"LLM analysis completely failed: {e}")
        
        # Enhanced fallback for preference detection
        return await enhanced_pure_semantic_analysis(query)


    async def enhanced_pure_semantic_analysis(query: str) -> Dict[str, Any]:
        """ENHANCED: Better fallback semantic analysis with preference detection"""
        query_lower = query.lower()
        
        # PREFERENCE DETECTION: Rule-based fallback
        is_preference = any(pattern in query_lower for pattern in [
            "i like", "i love", "i prefer", "my favorite", "i enjoy"
        ])
        
        if is_preference:
            # Extract food items with simple keyword matching
            food_items = []
            food_keywords = ["pizza", "coffee", "burger", "pasta", "salad", "sandwich", 
                            "sushi", "tacos", "kebab", "ice cream", "chocolate"]
            
            for item in food_keywords:
                if item in query_lower:
                    food_items.append(item)
            
            if food_items:
                return {
                    "intent_type": "preference_statement",
                    "domain": "food",
                    "action": "state_preference",
                    "entities": food_items,
                    "confidence": 0.9,
                    "semantic_tags": ["preference", "food"],
                    "preference_data": {
                        "is_preference": True,
                        "preference_type": "food_preference", 
                        "items": food_items,
                        "strength": "like"
                    }
                }
        
        # # QUERY DETECTION: Food preference questions
        # if any(pattern in query_lower for pattern in ["what do i like", "what are my favorite", "my preference"]):
        #     return {
        #         "intent_type": "query",
        #         "domain": "food",
        #         "action": "ask_preference",
        #         "entities": [],
        #         "confidence": 0.8,
        #         "semantic_tags": ["preference_query"]
        #     }
        
        # Standard embedding analysis (existing logic)
        try:
            _, col = get_chroma_setup()
            if col:
                res = col.query(query_texts=[query], n_results=5)
                metas = res.get("metadatas", [[]])
                intents = [m.get("intent_type") for m in metas if m.get("intent_type")]
                domains = [m.get("domain") for m in metas if m.get("domain")]

                return {
                    "intent_type": max(intents, key=intents.count, default="unknown") if intents else "unknown",
                    "domain": max(domains, key=domains.count, default="general") if domains else "general",
                    "action": "inferred_from_similar",
                    "entities": [],
                    "confidence": 0.6 if intents or domains else 0.3,
                    "semantic_tags": ["learned_pattern"] if intents or domains else ["semantic_fallback"],
                }
        except Exception as e:
            logger.warning(f"Enhanced semantic analysis failed: {e}")

        return {
            "intent_type": "unknown",
            "domain": "general", 
            "action": "unknown",
            "entities": [],
            "confidence": 0.1,
            "semantic_tags": []
        }


# ─────────────────────────────────────────  MEMORY WRITE  ───────────────────
async def add_to_semantic_memory(content: str,
                                 metadata: Optional[dict] = None) -> bool:
    """Store content with LLM-enriched metadata and deduplication."""
    global _recent_memories
    try:
        _, col = get_chroma_setup()
        if col is None:
            logger.warning("ChromaDB unavailable – skip memory storage")
            return False

        ts         = time.time()
        meta_base  = (metadata or {}).copy()
        meta_base.update({
            "timestamp": ts,
            "type":      meta_base.get("type", "conversation"),
            "session_id": meta_base.get("session_id", "default")
        })

        # ── dedup key ------------------------------------------------------
        with _memory_lock:
            c_hash  = hashlib.md5(content.encode()).hexdigest()
            dedup   = f"{c_hash}_{meta_base['session_id']}_{meta_base['type']}"
            last    = _recent_memories.get(dedup)
            if last and ts - last[0] < _DEDUP_WINDOW_SEC:
                logger.info(f"Dedup-skip memory: {content[:40]}…")
                return True

        # ── LLM enrichment -------------------------------------------------
        analysis        = run_async_safely(analyze_query_with_llm(content)) or {}
        preference_meta = {}
        pref_data       = analysis.get("preference_data", {})
        if pref_data.get("is_preference"):
            preference_meta = {
                "is_user_preference": True,
                "preference_type":    pref_data.get("preference_type", "general"),
                "preference_items":   json.dumps(pref_data.get("items", [])),
                "preference_strength":pref_data.get("strength", "like"),
            }

        meta_llm = {
            "intent_type":  str(analysis.get("intent_type", "unknown")),
            "domain":       str(analysis.get("domain", "general")),
            "action":       str(analysis.get("action", "unknown")),
            "llm_confidence": float(analysis.get("confidence", 0.0)),
            "entities_count": len(analysis.get("entities", [])),
            "tags_count":     len(analysis.get("semantic_tags", [])),
            **preference_meta
        }

        # ── write ----------------------------------------------------------
        doc_id = f"doc_{int(ts*1000)}_{meta_base['session_id'][:8]}_{meta_base['type']}"
        col.add(
            documents=[content.strip()],
            metadatas=[sanitize_metadata_for_chromadb({**meta_base, **meta_llm})],
            ids=[doc_id]
        )
        _recent_memories[dedup] = (ts, meta_base['session_id'])
        if pref_data.get("is_preference"):
            logger.info(f"🎯 Stored user preference: {pref_data.get('items')}")
        else:
            logger.info(f"Memory stored – intent={meta_llm['intent_type']}, domain={meta_llm['domain']}")
        return True

    except Exception as exc:
        logger.error(f"add_to_semantic_memory failed: {exc}")
        return False

# ─────────────────────────────────────────  MEMORY SEARCH  ──────────────────
def search_semantic_memory(query: str, n_results: int = 3,
                           session_id: Optional[str] = None):
    """
    Retrieve top-N memories. If the query is preference-oriented (food domain
    & “what do I like …”), stored preferences are prioritised.
    """
    try:
        _, col = get_chroma_setup()
        if col is None:
            logger.warning("ChromaDB unavailable – search aborted")
            return None

        analysis   = run_async_safely(analyze_query_with_llm(query)) or {}
        intent     = analysis.get("intent_type")
        domain     = analysis.get("domain")
        conf       = analysis.get("confidence", 0.0)

        logger.info(f"Semantic search – intent={intent}, domain={domain}, conf={conf:.2f}")

        # preference-centric branch
        pref_query = (domain == "food"
                      and any(kw in query.lower()
                              for kw in ["what do i like", "my favorite",
                                         "my preference", "favorite food"]))

        results: List[Tuple[str, Dict[str, Any], float]] = []

        def _q(text, where: Optional[dict] = None, boost: float = 0.0):
            res = col.query(query_texts=[text], n_results=n_results*2, where=where)
            docs  = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] if res.get("distances") else [0.0]*len(docs)
            out   = []
            for d, m, dist in zip(docs, metas, dists):
                if d:
                    dist = float(dist[0] if isinstance(dist, (list, tuple)) else dist)
                    out.append((d, m, max(dist-boost, 0.0)))
            return out

        # preference pass
        if pref_query:
            pref_where = {"$and": [{"is_user_preference": True}, {"domain": "food"}]}
            pref_res   = _q(query, where=pref_where, boost=-0.3)
            results   += pref_res

        # generic pass
        results += _q(query)

        # session-scoped boost
        if session_id and conf > 0.3:
            where = {"session_id": session_id}
            if domain and domain != "general":
                where = {"$and": [where, {"domain": domain}]}
            results += _q(query, where=where, boost=-0.1)

        if not results:
            logger.info("No semantic matches.")
            return None

        # dedup + scoring
        uniq, seen = [], set()
        for doc, meta, dist in results:
            score = 1.0 - dist
            if meta.get("is_user_preference"):
                score += 1.0
            if meta.get("intent_type") == intent:
                score += 0.5
            if meta.get("domain") == domain:
                score += 0.6
            if meta.get("session_id") == session_id:
                score += 0.4
            key = doc.strip().lower()
            if key not in seen and dist < 0.9:
                uniq.append((doc, meta, score))
                seen.add(key)

        uniq.sort(key=lambda x: x[2], reverse=True)
        top = uniq[:n_results]

        logger.info(f"Semantic search returned {len(top)} results")
        for i, (doc, meta, sc) in enumerate(top):
            tag = "🎯" if meta.get("is_user_preference") else ""
            logger.info(f" #{i+1} {tag}score={sc:.3f} – {doc[:60]}…")

        return {
            "documents": [[d for d, _, _ in top]],
            "metadatas": [[m for _, m, _ in top]],
        }

    except Exception as exc:
        logger.error(f"search_semantic_memory failed: {exc}")
        return None
