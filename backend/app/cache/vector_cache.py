import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.app.core.logging import logger
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.services.rag_service import (
    HybridSearchIndex,
    MultiEngineHybridIndex,
    rag_service,
)


class VectorStoreCache:
    """Persistent on-disk and in-memory cache for active vector stores and document metadata."""

    def __init__(
        self,
        storage_dir: str = "data/vector_storage",
        ttl_seconds: float = 86400.0,
    ):
        self.storage_dir = storage_dir
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        os.makedirs(self.storage_dir, exist_ok=True)

    @staticmethod
    def compute_hash(text: str) -> str:
        """Generates a consistent 16-character hex hash from URL, hash string, or text."""
        try:
            clean = (text or "").strip()
            if re.fullmatch(r"[a-f0-9]{16,64}", clean):
                return clean[:16]
            return compute_url_hash(clean)[:16]
        except Exception as e:
            logger.error(f"VectorStoreCache: Error computing hash: {e}", exc_info=True)
            return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:16]

    def _get_item_dir(self, cache_key: str) -> str:
        """Returns the filesystem directory path for a specific cache key."""
        try:
            return os.path.join(self.storage_dir, cache_key)
        except Exception as e:
            logger.error(f"VectorStoreCache: Error getting item directory: {e}", exc_info=True)
            return f"data/vector_storage/{cache_key}"

    def _persist_to_disk(
        self,
        cache_key: str,
        vector_store: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Persists vector store index, metadata, and chunk documents to disk."""
        try:
            item_dir = self._get_item_dir(cache_key)
            faiss_dir = os.path.join(item_dir, "faiss")
            os.makedirs(faiss_dir, exist_ok=True)

            docs_payload: List[Dict[str, Any]] = []
            collection_name = getattr(vector_store, "collection_name", f"doc_{cache_key}")
            vec_sid = getattr(vector_store, "session_id", None)

            meta_copy = dict(metadata or {})
            if vec_sid and not meta_copy.get("vector_session_id"):
                meta_copy["vector_session_id"] = vec_sid

            # Safely save FAISS index if available and non-None
            inner_vs = getattr(vector_store, "vector_store", None)
            if inner_vs is not None and hasattr(inner_vs, "save_local"):
                inner_vs.save_local(faiss_dir)
            elif isinstance(vector_store, FAISS):
                vector_store.save_local(faiss_dir)

            # Safely collect documents payload
            docs = getattr(vector_store, "documents", None)
            if docs and isinstance(docs, list):
                for doc in docs:
                    docs_payload.append({
                        "page_content": getattr(doc, "page_content", ""),
                        "metadata": getattr(doc, "metadata", {}),
                    })

            meta_file = os.path.join(item_dir, "meta.json")
            payload = {
                "cache_key": cache_key,
                "collection_name": collection_name,
                "created_at": time.time(),
                "metadata": meta_copy,
                "documents": docs_payload,
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"VectorStoreCache: Successfully persisted index '{cache_key}' to disk at {item_dir}")
        except Exception as e:
            logger.error(f"VectorStoreCache: Failed to persist '{cache_key}' to disk: {e}", exc_info=True)

    def _load_from_disk(self, cache_key: str) -> Optional[Any]:
        """Restores vector store index and metadata from disk storage."""
        try:
            item_dir = self._get_item_dir(cache_key)
            faiss_dir = os.path.join(item_dir, "faiss")
            meta_file = os.path.join(item_dir, "meta.json")

            if not os.path.exists(meta_file):
                return None

            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_docs = data.get("documents", [])
            metadata = data.get("metadata", {})
            collection_name = data.get("collection_name", f"doc_{cache_key}")
            target_sid = metadata.get("vector_session_id") or metadata.get("session_id")

            faiss_vs = None
            if os.path.exists(faiss_dir) and os.path.exists(os.path.join(faiss_dir, "index.faiss")):
                try:
                    faiss_vs = FAISS.load_local(
                        folder_path=faiss_dir,
                        embeddings=rag_service.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except Exception as fe:
                    logger.warning(f"VectorStoreCache: Could not load FAISS local index: {fe}")

            if faiss_vs and raw_docs:
                documents = [
                    Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                    for d in raw_docs
                ]
                bm25_retriever = BM25Retriever.from_documents(documents, k=len(documents))
                faiss_hybrid = HybridSearchIndex(
                    vector_store=faiss_vs,
                    bm25_retriever=bm25_retriever,
                    documents=documents,
                )
                final_index = MultiEngineHybridIndex(
                    collection_name=collection_name,
                    faiss_hybrid_index=faiss_hybrid,
                    documents=documents,
                    embeddings=rag_service.embeddings,
                    session_id=target_sid,
                    metadata=metadata,
                )
            elif target_sid:
                final_index = rag_service.get_session_index(target_sid, metadata=metadata)
            elif faiss_vs:
                final_index = faiss_vs
            else:
                return None

            self._cache[cache_key] = {
                "vector_store": final_index,
                "metadata": metadata,
                "created_at": data.get("created_at", time.time()),
                "last_accessed": time.time(),
            }
            logger.info(f"VectorStoreCache: Restored index '{cache_key}' from disk storage.")
            return final_index
        except Exception as e:
            logger.error(f"VectorStoreCache: Error restoring '{cache_key}' from disk: {e}", exc_info=True)
            return None

    def set(self, key_source: str, vector_store: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Stores a vectorstore or hybrid index in RAM and persists it to disk."""
        try:
            cache_key = self.compute_hash(key_source)
            with self._lock:
                self._cache[cache_key] = {
                    "vector_store": vector_store,
                    "metadata": metadata or {},
                    "created_at": time.time(),
                    "last_accessed": time.time(),
                }
            self._persist_to_disk(cache_key, vector_store, metadata)
            logger.info(f"VectorStoreCache: Stored index for key '{cache_key}'")
            return cache_key
        except Exception as e:
            logger.error(f"VectorStoreCache: Error setting vector store: {e}", exc_info=True)
            return ""

    def get(self, key_source: str) -> Optional[Any]:
        """Retrieves vectorstore or hybrid index, checking RAM first then loading from disk."""
        try:
            cache_key = self.compute_hash(key_source)
            with self._lock:
                entry = self._cache.get(cache_key)
                if entry:
                    entry["last_accessed"] = time.time()
                    return entry["vector_store"]

            loaded = self._load_from_disk(cache_key)
            return loaded
        except Exception as e:
            logger.error(f"VectorStoreCache: Error getting vector store: {e}", exc_info=True)
            return None

    def get_metadata(self, key_source: str) -> Optional[Dict[str, Any]]:
        """Retrieves stored metadata for the key source from RAM or disk."""
        try:
            cache_key = self.compute_hash(key_source)
            with self._lock:
                entry = self._cache.get(cache_key)
                if entry:
                    return entry.get("metadata")

            item_dir = self._get_item_dir(cache_key)
            meta_file = os.path.join(item_dir, "meta.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("metadata")
                except Exception as e:
                    logger.warning(f"VectorStoreCache: Failed reading metadata file for '{cache_key}': {e}")
            return None
        except Exception as e:
            logger.error(f"VectorStoreCache: Error getting metadata: {e}", exc_info=True)
            return None

    def clear(self):
        """Purges RAM cache."""
        try:
            with self._lock:
                self._cache.clear()
                logger.info("VectorStoreCache: Purged all in-memory entries.")
        except Exception as e:
            logger.error(f"VectorStoreCache: Error clearing cache: {e}", exc_info=True)

    def count(self) -> int:
        """Returns the number of active vector stores in memory."""
        try:
            with self._lock:
                return len(self._cache)
        except Exception as e:
            logger.error(f"VectorStoreCache: Error counting entries: {e}", exc_info=True)
            return 0


vector_store_cache = VectorStoreCache()
