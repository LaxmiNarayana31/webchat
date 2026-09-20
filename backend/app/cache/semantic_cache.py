import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
from upstash_redis import Redis as UpstashRedis

from backend.app.clients.gemini_client import gemini_client
from backend.app.core.config import settings
from backend.app.core.logging import logger


class SemanticCacheService:
    """Production-grade Semantic Cache powered by Upstash Cloud Redis.

    Implements a two-tier semantic caching pattern:
      - Tier 1: Instant exact-match lookup (SHA-256 hash over normalized query + context)
      - Tier 2: Dense vector similarity search using Google Gemini embeddings and
                vectorized cosine distance against cached queries partitioned by context (URL).
    Includes automatic TTL expiration, local on-disk fallback for offline resilience,
    and multi-tenant document isolation.
    """

    def __init__(
        self,
        storage_dir: str = "data/semantic_cache",
        ttl_seconds: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
    ):
        self.storage_dir = storage_dir
        self.ttl_seconds = int(ttl_seconds or settings.SEMANTIC_CACHE_TTL)
        self.similarity_threshold = float(
            similarity_threshold if similarity_threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD
        )
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._redis_client: Optional[UpstashRedis] = None
        self._redis_available: bool = False

        os.makedirs(self.storage_dir, exist_ok=True)
        self._init_redis()
        self._load_local_disk_cache()

    def _init_redis(self):
        """Initializes connection to Upstash Redis using REDIS_URL or REST credentials."""
        try:
            redis_url = settings.REDIS_URL or os.getenv("REDIS_URL", "")
            rest_url = settings.UPSTASH_REDIS_REST_URL or os.getenv("UPSTASH_REDIS_REST_URL", "")
            rest_token = settings.UPSTASH_REDIS_REST_TOKEN or os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

            # Mode 1: Explicit REST URL and token
            if rest_url and rest_token:
                self._redis_client = UpstashRedis(url=rest_url.strip(), token=rest_token.strip())
                ping_res = self._redis_client.ping()
                self._redis_available = True
                logger.info(f"SemanticCache: Connected to Upstash Redis REST API ({rest_url}) - ping={ping_res}")
                return

            # Mode 2: Parse rediss:// or redis:// URL into Upstash REST endpoint
            if redis_url and redis_url.strip():
                clean_url = redis_url.strip()
                parsed = urllib.parse.urlparse(clean_url)
                hostname = parsed.hostname
                token = parsed.password

                if hostname and token:
                    http_url = f"https://{hostname}"
                    self._redis_client = UpstashRedis(url=http_url, token=token)
                    ping_res = self._redis_client.ping()
                    self._redis_available = True
                    logger.info(
                        f"SemanticCache: Connected to Upstash Redis at {hostname} (ping={ping_res}, TTL={self.ttl_seconds}s, threshold={self.similarity_threshold})"
                    )
                    return

            logger.warning("SemanticCache: REDIS_URL not configured. Operating in local-only mode.")
            self._redis_available = False
        except Exception as e:
            logger.warning(
                f"SemanticCache: Could not connect to Upstash Redis ({e}). Gracefully falling back to local storage.",
                exc_info=False,
            )
            self._redis_available = False

    @property
    def is_redis_active(self) -> bool:
        """Returns True if Upstash Redis connection is established and healthy."""
        return self._redis_available and self._redis_client is not None

    def _normalize_query(self, query: str) -> str:
        """Normalizes query string for uniform exact hashing and similarity indexing."""
        try:
            clean = (query or "").lower().strip()
            clean = re.sub(r"\s+", " ", clean)
            clean = re.sub(r"[^\w\s\?]", "", clean)
            return clean
        except Exception as e:
            logger.error(f"SemanticCache: Error normalizing query: {e}")
            return str(query).strip().lower() if query else ""

    def _compute_exact_key(self, query: str, context_hash: Optional[str] = None) -> str:
        """Computes SHA-256 hash for Tier 1 instant exact match."""
        clean = self._normalize_query(query)
        payload = f"{context_hash or 'global'}:{clean}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """Embeds query using Google Gemini embedding model."""
        try:
            clean = query.strip()
            if not clean:
                return None
            embs = gemini_client.embed_texts([clean])
            if embs and len(embs) > 0 and len(embs[0]) > 0:
                return embs[0]
            return None
        except Exception as e:
            logger.warning(f"SemanticCache: Failed generating query embedding: {e}")
            return None

    def _get_item_path(self, cache_key: str) -> str:
        """Returns filesystem path for local disk persistence."""
        return os.path.join(self.storage_dir, f"{cache_key}.json")

    def _load_local_disk_cache(self):
        """Loads semantic cache entries from disk on startup for offline resilience."""
        try:
            count = 0
            now = time.time()
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if now - data.get("created_at", 0) > self.ttl_seconds:
                                os.remove(filepath)
                                continue
                            self._local_cache[data.get("cache_key", filename[:-5])] = data
                            count += 1
                    except Exception as e:
                        logger.debug(f"SemanticCache: Skipped local file {filename}: {e}")
            if count > 0:
                logger.info(f"SemanticCache: Loaded {count} local entries from disk.")
        except Exception as e:
            logger.error(f"SemanticCache: Error loading from disk: {e}")

    def _persist_to_local_disk(self, cache_key: str, payload: Dict[str, Any]):
        """Saves entry to local disk."""
        try:
            item_path = self._get_item_path(cache_key)
            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"SemanticCache: Failed persisting {cache_key} to local disk: {e}")

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two 1D dense vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def get_cached_answer(
        self,
        query: str,
        context_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cached answer via Tier 1 (Exact Match) or Tier 2 (Dense Vector Search).

        Args:
            query: The user prompt or question.
            context_hash: Active URL or document identifier ensuring context isolation.

        Returns:
            Dictionary with answer, citations, similarity score, tier, and matched query if hit, else None.
        """
        if not settings.SEMANTIC_CACHE_ENABLED:
            return None

        clean_query = self._normalize_query(query)
        if not clean_query:
            return None

        context_scope = context_hash or "global"
        exact_key = self._compute_exact_key(clean_query, context_scope)
        start_time = time.time()

        # =========================================================================
        # TIER 1: Instant Exact Match (SHA-256 Lookup in Upstash Redis)
        # =========================================================================
        if self.is_redis_active and self._redis_client:
            try:
                exact_redis_key = f"sem_cache:exact:{exact_key}"
                entry_id = self._redis_client.get(exact_redis_key)

                if entry_id:
                    entry_json = self._redis_client.get(f"sem_cache:entry:{entry_id}")
                    if entry_json:
                        data = json.loads(entry_json)
                        elapsed_ms = (time.time() - start_time) * 1000
                        logger.info(
                            f"SemanticCache: HIT (Tier 1 Exact Match) in {elapsed_ms:.1f}ms for query '{query[:40]}...'"
                        )
                        return {
                            "answer": data.get("answer", ""),
                            "citations": data.get("citations", []),
                            "telemetry": data.get("telemetry", {}),
                            "similarity": 1.0,
                            "tier": "exact",
                            "cached": True,
                            "matched_query": data.get("original_query", query),
                        }
            except Exception as e:
                logger.warning(f"SemanticCache: Redis Tier 1 exact match check failed: {e}")

        # Local Tier 1 Exact Match Check
        with self._lock:
            local_entry = self._local_cache.get(exact_key)
            if local_entry and (time.time() - local_entry.get("created_at", 0) <= self.ttl_seconds):
                logger.info(f"SemanticCache: HIT (Local Tier 1 Exact Match) for query '{query[:40]}...'")
                return {
                    "answer": local_entry.get("answer", ""),
                    "citations": local_entry.get("citations", []),
                    "telemetry": local_entry.get("telemetry", {}),
                    "similarity": 1.0,
                    "tier": "exact",
                    "cached": True,
                    "matched_query": local_entry.get("original_query", query),
                }

        # =========================================================================
        # TIER 2: Dense Vector Semantic Similarity Search
        # =========================================================================
        query_vec = self._embed_query(query)
        if query_vec is None:
            return None

        q_np = np.array(query_vec, dtype=np.float32)

        # 1. Search in Upstash Redis Partition
        if self.is_redis_active and self._redis_client:
            try:
                index_key = f"sem_cache:idx:{context_scope}"
                candidate_ids = self._redis_client.smembers(index_key)

                if candidate_ids and isinstance(candidate_ids, (list, set)):
                    best_score = -1.0
                    best_entry = None

                    # Iterate candidates (capped at 60 most relevant for sub-millisecond scoring)
                    id_list = list(candidate_ids)[:60]
                    for cid in id_list:
                        raw = self._redis_client.get(f"sem_cache:entry:{cid}")
                        if not raw:
                            continue
                        cand_data = json.loads(raw)
                        c_vec = cand_data.get("vector")
                        if not c_vec or not isinstance(c_vec, list):
                            continue

                        cand_np = np.array(c_vec, dtype=np.float32)
                        score = self._cosine_similarity(q_np, cand_np)

                        if score > best_score:
                            best_score = score
                            best_entry = cand_data

                    if best_entry and best_score >= self.similarity_threshold:
                        elapsed_ms = (time.time() - start_time) * 1000
                        logger.info(
                            f"SemanticCache: HIT (Tier 2 Semantic Score: {best_score:.4f} >= {self.similarity_threshold}) "
                            f"in {elapsed_ms:.1f}ms for query '{query[:40]}...'"
                        )
                        return {
                            "answer": best_entry.get("answer", ""),
                            "citations": best_entry.get("citations", []),
                            "telemetry": best_entry.get("telemetry", {}),
                            "similarity": round(best_score, 4),
                            "tier": "semantic",
                            "cached": True,
                            "matched_query": best_entry.get("original_query", ""),
                        }
                    elif best_score > 0:
                        logger.info(
                            f"SemanticCache: MISS (best score {best_score:.4f} < threshold {self.similarity_threshold}) "
                            f"for query '{query[:40]}...'"
                        )
            except Exception as e:
                logger.warning(f"SemanticCache: Redis Tier 2 vector search error: {e}")

        # 2. Local Fallback Vector Search
        with self._lock:
            best_local_score = -1.0
            best_local_entry = None
            now = time.time()

            for item in self._local_cache.values():
                if item.get("context_hash") != context_scope:
                    continue
                if now - item.get("created_at", 0) > self.ttl_seconds:
                    continue
                c_vec = item.get("vector")
                if not c_vec:
                    continue

                cand_np = np.array(c_vec, dtype=np.float32)
                score = self._cosine_similarity(q_np, cand_np)
                if score > best_local_score:
                    best_local_score = score
                    best_local_entry = item

            if best_local_entry and best_local_score >= self.similarity_threshold:
                logger.info(
                    f"SemanticCache: HIT (Local Tier 2 Semantic Score: {best_local_score:.4f} >= {self.similarity_threshold}) "
                    f"for query '{query[:40]}...'"
                )
                return {
                    "answer": best_local_entry.get("answer", ""),
                    "citations": best_local_entry.get("citations", []),
                    "telemetry": best_local_entry.get("telemetry", {}),
                    "similarity": round(best_local_score, 4),
                    "tier": "semantic",
                    "cached": True,
                    "matched_query": best_local_entry.get("original_query", ""),
                }

        return None

    def set_cached_answer(
        self,
        query: str,
        answer: str,
        citations: list,
        context_hash: Optional[str] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        query_vector: Optional[List[float]] = None,
    ):
        """Stores query, embedding vector, answer, and citations in Upstash Redis and local disk."""
        if not settings.SEMANTIC_CACHE_ENABLED:
            return

        clean_query = self._normalize_query(query)
        if not clean_query or not answer:
            return

        context_scope = context_hash or "global"
        exact_key = self._compute_exact_key(clean_query, context_scope)
        entry_id = str(uuid.uuid4())[:16]

        # Generate embedding vector if not provided
        vec = query_vector or self._embed_query(query)

        payload = {
            "entry_id": entry_id,
            "cache_key": exact_key,
            "original_query": query,
            "normalized_query": clean_query,
            "context_hash": context_scope,
            "answer": answer,
            "citations": citations or [],
            "telemetry": telemetry or {},
            "vector": vec,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }

        # 1. Store in Upstash Cloud Redis
        if self.is_redis_active and self._redis_client:
            try:
                # Save entry payload
                self._redis_client.set(
                    f"sem_cache:entry:{entry_id}",
                    json.dumps(payload, ensure_ascii=False),
                    ex=self.ttl_seconds,
                )
                # Map exact key to entry ID
                self._redis_client.set(
                    f"sem_cache:exact:{exact_key}",
                    entry_id,
                    ex=self.ttl_seconds,
                )
                # Register in partition index for context_hash
                index_key = f"sem_cache:idx:{context_scope}"
                self._redis_client.sadd(index_key, entry_id)
                self._redis_client.expire(index_key, self.ttl_seconds)

                logger.info(
                    f"SemanticCache: Stored answer in Upstash Redis for query '{query[:35]}...' (TTL: {self.ttl_seconds}s)"
                )
            except Exception as e:
                logger.warning(f"SemanticCache: Failed storing in Upstash Redis: {e}")

        # 2. Store in local RAM & Disk storage
        with self._lock:
            self._local_cache[exact_key] = payload
        self._persist_to_local_disk(exact_key, payload)

    def clear(self, context_hash: Optional[str] = None):
        """Purges cached entries for a specific context or completely purges all entries."""
        context_scope = context_hash or "global"

        if self.is_redis_active and self._redis_client:
            try:
                if context_hash:
                    index_key = f"sem_cache:idx:{context_scope}"
                    cids = self._redis_client.smembers(index_key)
                    if cids:
                        for cid in cids:
                            self._redis_client.delete(f"sem_cache:entry:{cid}")
                    self._redis_client.delete(index_key)
                    logger.info(f"SemanticCache: Purged Upstash Redis cache for context '{context_scope}'")
                else:
                    logger.info("SemanticCache: Purging all Redis semantic cache entries...")
            except Exception as e:
                logger.warning(f"SemanticCache: Redis clear warning: {e}")

        # Clear local cache
        with self._lock:
            if context_hash:
                to_delete = [k for k, v in self._local_cache.items() if v.get("context_hash") == context_scope]
                for k in to_delete:
                    del self._local_cache[k]
                    try:
                        os.remove(self._get_item_path(k))
                    except Exception:
                        pass
            else:
                self._local_cache.clear()
                for filename in os.listdir(self.storage_dir):
                    if filename.endswith(".json"):
                        try:
                            os.remove(os.path.join(self.storage_dir, filename))
                        except Exception:
                            pass
                logger.info("SemanticCache: Purged all local cache entries.")


semantic_cache_service = SemanticCacheService()
semantic_cache = semantic_cache_service