import hashlib
import math
import re
from typing import Any, Dict, List, Optional
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    PointStruct,
    Prefetch,
    QuantizationSearchParams,
    Rrf,
    RrfQuery,
    SearchParams,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.app.core.logging import logger
from backend.app.core.config import settings

load_dotenv()


class SparseBM25Vectorizer:
    """Computes sparse lexical vector representations for Qdrant sparse search."""

    MAX_DIM = 2**30

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenizes input string into lowercase alphanumeric words."""
        try:
            if not text:
                return []
            return [w.lower() for w in re.findall(r"\b\w{2,}\b", str(text))]
        except Exception as e:
            logger.error(f"Error in tokenize: {e}", exc_info=True)
            return []

    @classmethod
    def hash_token(cls, token: str) -> int:
        """Hashes token into deterministic 32-bit integer index."""
        try:
            return int(hashlib.md5(str(token).encode("utf-8")).hexdigest()[:8], 16) % cls.MAX_DIM
        except Exception as e:
            logger.error(f"Error in hash_token: {e}", exc_info=True)
            return 0

    @classmethod
    def transform(cls, text: str) -> SparseVector:
        """Transforms text into sparse frequency-weighted vector."""
        try:
            tokens = cls.tokenize(text)
            if not tokens:
                return SparseVector(indices=[], values=[])

            tf_counts: Dict[int, float] = {}
            for token in tokens:
                idx = cls.hash_token(token)
                tf_counts[idx] = tf_counts.get(idx, 0.0) + 1.0

            indices: List[int] = []
            values: List[float] = []

            for idx, count in tf_counts.items():
                # Sub-linear term frequency scaling: 1 + ln(count)
                weight = 1.0 + math.log(count)
                indices.append(idx)
                values.append(float(weight))

            return SparseVector(indices=indices, values=values)
        except Exception as e:
            logger.error(f"Error in transform: {e}", exc_info=True)
            return SparseVector(indices=[], values=[])


class QdrantService:
    """Manages Qdrant Cloud vector database operations, quantization, and hybrid search."""

    def __init__(self):
        """Initializes Qdrant client placeholder without local disk storage."""
        try:
            self._client: Optional[QdrantClient] = None
        except Exception as e:
            logger.error(f"Error initializing QdrantService: {e}", exc_info=True)
            raise

    @property
    def client(self) -> QdrantClient:
        """Initializes and returns configured Cloud QdrantClient instance. Qdrant is strictly mandatory."""
        try:
            if self._client is not None:
                return self._client

            qdrant_url = settings.QDRANT_URL
            qdrant_api_key = settings.QDRANT_API_KEY

            if not qdrant_url or not qdrant_url.strip():
                logger.error("QdrantService: QDRANT_URL is not set. Qdrant Cloud is strictly mandatory.")
                raise RuntimeError("QDRANT_URL is not configured in environment. Qdrant is strictly mandatory.")

            clean_url = qdrant_url.strip()
            logger.info(f"QdrantService: Connecting to mandatory Cloud Qdrant cluster at {clean_url}")
            self._client = QdrantClient(
                url=clean_url,
                api_key=qdrant_api_key.strip() if qdrant_api_key else None,
                timeout=20,
                prefer_grpc=False,
            )
            return self._client
        except Exception as e:
            logger.error(f"QdrantService: Failed initializing Cloud Qdrant: {e}", exc_info=True)
            raise RuntimeError(f"Failed to connect to mandatory Qdrant cluster: {e}") from e

    def create_hybrid_collection(
        self,
        collection_name: str,
        vector_dim: int = 3072,
    ) -> bool:
        """Creates Qdrant collection configured for dense, sparse, and quantized vectors."""
        try:
            if self.client is None:
                return False

            existing = [c.name for c in self.client.get_collections().collections]
            if collection_name not in existing:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=vector_dim,
                            distance=Distance.COSINE,
                            on_disk=True,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=True)
                        )
                    },
                    quantization_config=BinaryQuantization(
                        binary=BinaryQuantizationConfig(always_ram=True)
                    ),
                )
                logger.info(f"QdrantService: Created collection '{collection_name}' with Binary Quantization.")

            # Ensure payload index on session_id for high-performance session filtering
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="session_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"QdrantService: Error creating collection '{collection_name}': {e}", exc_info=True)
            return False

    def insert_hybrid_chunks(
        self,
        collection_name: str,
        texts: List[str],
        dense_embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Inserts dense and sparse vector points into Qdrant collection partitioned by session_id."""
        try:
            if self.client is None:
                return False

            self.create_hybrid_collection(collection_name, vector_dim=len(dense_embeddings[0]))

            points: List[PointStruct] = []
            for i, (text, dense_vec) in enumerate(zip(texts, dense_embeddings)):
                sparse_vec = SparseBM25Vectorizer.transform(text)
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                payload = {
                    "text": text,
                    "chunk_index": meta.get("chunk_index", i),
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "session_id": meta.get("session_id", ""),
                    "parent_id": meta.get("parent_id", ""),
                    "parent_text": meta.get("parent_text", text),
                }

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vec,
                            "sparse": sparse_vec,
                        },
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            logger.info(f"QdrantService: Upserted {len(points)} hybrid points into '{collection_name}' (session: {metadatas[0].get('session_id') if metadatas else 'none'}).")
            return True
        except Exception as e:
            logger.error(f"QdrantService: Failed inserting points into '{collection_name}': {e}", exc_info=True)
            return False

    def search_hybrid(
        self,
        collection_name: str,
        query_text: str,
        query_dense_vector: List[float],
        top_k: int = 4,
        filter_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Executes hybrid dense and sparse search with reciprocal rank fusion."""
        try:
            if self.client is None:
                return []

            query_sparse_vector = SparseBM25Vectorizer.transform(query_text)
            prefetch_limit = max(top_k * 3, 12)

            query_filter = None
            if filter_session_id:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=filter_session_id),
                        )
                    ]
                )

            prefetch_queries = [
                Prefetch(
                    query=query_dense_vector,
                    using="dense",
                    filter=query_filter,
                    params=SearchParams(
                        quantization=QuantizationSearchParams(rescore=True, oversampling=2.0)
                    ),
                    limit=prefetch_limit,
                ),
            ]

            if query_sparse_vector.indices:
                prefetch_queries.append(
                    Prefetch(
                        query=query_sparse_vector,
                        using="sparse",
                        filter=query_filter,
                        limit=prefetch_limit,
                    )
                )

            response = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_queries,
                query=RrfQuery(rrf=Rrf(k=60)),
                limit=top_k,
            )

            results: List[Dict[str, Any]] = []
            for point in response.points:
                payload = point.payload or {}
                results.append({
                    "content": payload.get("text", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "parent_id": payload.get("parent_id", ""),
                    "parent_text": payload.get("parent_text", payload.get("text", "")),
                    "url": payload.get("url", ""),
                    "title": payload.get("title", ""),
                    "session_id": payload.get("session_id", ""),
                    "score": point.score,
                })

            return results
        except Exception as e:
            logger.warning(f"QdrantService: Hybrid search on '{collection_name}' error: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Deletes a collection from Qdrant."""
        try:
            if self.client is None:
                return False
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"QdrantService: Deleted collection '{collection_name}'.")
            return True
        except Exception as e:
            logger.warning(f"QdrantService: Error deleting collection '{collection_name}': {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Checks if a collection exists in Qdrant."""
        try:
            if self.client is None:
                return False
            existing = [c.name for c in self.client.get_collections().collections]
            return collection_name in existing
        except Exception:
            return False


qdrant_service = QdrantService()
