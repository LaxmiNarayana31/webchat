import json
import time
from typing import Any, Dict, Optional

from backend.app.core.logging import logger
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.models.models import UrlCacheEntity
from backend.config.database import get_db_session


class UrlCacheRepository:
    """Database-backed repository for caching ingested URLs, extracted metadata, and vector references."""

    def get_by_hash(self, url_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached URL record by its hash and updates access statistics."""
        try:
            if not url_hash:
                return None

            with get_db_session() as session:
                entity = (
                    session.query(UrlCacheEntity)
                    .filter(UrlCacheEntity.url_hash == url_hash)
                    .first()
                )
                if not entity:
                    return None

                # Update access statistics
                entity.last_accessed_at = time.time()  # type: ignore
                entity.access_count = (entity.access_count or 0) + 1  # type: ignore
                session.flush()

                logger.info(
                    f"UrlCacheRepository: Cache HIT for hash '{url_hash[:12]}' -> "
                    f"'{entity.title}' (accesses: {entity.access_count}, vector_session_id: '{entity.vector_session_id}')"
                )

                return {
                    "url_hash": entity.url_hash,
                    "url": entity.url,
                    "title": entity.title,
                    "vector_session_id": entity.vector_session_id,
                    "word_count": entity.word_count,
                    "strategy_used": entity.strategy_used,
                    "paywall_bypassed": entity.paywall_bypassed,
                    "metadata": entity.site_metadata,
                    "content": entity.content,
                    "created_at": entity.created_at,
                    "last_accessed_at": entity.last_accessed_at,
                    "access_count": entity.access_count,
                }
        except Exception as e:
            logger.error(f"UrlCacheRepository: Error retrieving cached hash '{url_hash}': {e}", exc_info=True)
            raise

    def get_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Computes URL hash and retrieves the corresponding cached record."""
        try:
            if not url:
                return None
            url_hash = compute_url_hash(url)
            return self.get_by_hash(url_hash)
        except Exception as e:
            logger.error(f"UrlCacheRepository: Error retrieving URL '{url}': {e}", exc_info=True)
            raise

    def save_url_cache(
        self,
        url_hash: str,
        url: str,
        title: str,
        vector_session_id: str,
        word_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
        strategy_used: str = "auto",
        paywall_bypassed: bool = False,
    ) -> Dict[str, Any]:
        """Persists or updates an ingested URL record and its vector reference in the database."""
        try:
            now = time.time()
            meta_dict = metadata or {}
            meta_json = json.dumps(meta_dict, default=str)

            with get_db_session() as session:
                entity = (
                    session.query(UrlCacheEntity)
                    .filter(UrlCacheEntity.url_hash == url_hash)
                    .first()
                )

                if entity:
                    entity.url = url  # type: ignore
                    entity.title = title[:250] if title else "Website Content"  # type: ignore
                    entity.vector_session_id = vector_session_id  # type: ignore
                    entity.word_count = word_count  # type: ignore
                    entity.strategy_used = strategy_used  # type: ignore
                    entity.paywall_bypassed = paywall_bypassed  # type: ignore
                    entity.metadata_json = meta_json  # type: ignore
                    if content:
                        entity.content = content  # type: ignore
                    entity.last_accessed_at = now  # type: ignore
                    entity.access_count = (entity.access_count or 0) + 1  # type: ignore
                else:
                    entity = UrlCacheEntity(
                        url_hash=url_hash,
                        url=url,
                        title=title[:250] if title else "Website Content",
                        vector_session_id=vector_session_id,
                        word_count=word_count,
                        strategy_used=strategy_used,
                        paywall_bypassed=paywall_bypassed,
                        metadata_json=meta_json,
                        content=content,
                        created_at=now,
                        last_accessed_at=now,
                        access_count=1,
                    )
                    session.add(entity)

                session.flush()

                logger.info(
                    f"UrlCacheRepository: Saved cache entry for hash '{url_hash[:12]}' -> "
                    f"'{entity.title}' (vector_session_id: '{vector_session_id}')"
                )

                return {
                    "url_hash": entity.url_hash,
                    "url": entity.url,
                    "title": entity.title,
                    "vector_session_id": entity.vector_session_id,
                    "word_count": entity.word_count,
                    "strategy_used": entity.strategy_used,
                    "paywall_bypassed": entity.paywall_bypassed,
                    "metadata": meta_dict,
                    "content": entity.content,
                    "created_at": entity.created_at,
                    "last_accessed_at": entity.last_accessed_at,
                    "access_count": entity.access_count,
                }
        except Exception as e:
            logger.error(f"UrlCacheRepository: Error saving cache entry for hash '{url_hash}': {e}", exc_info=True)
            raise

    def delete_by_hash(self, url_hash: str) -> bool:
        """Deletes a cached URL record from the database."""
        try:
            with get_db_session() as session:
                deleted = (
                    session.query(UrlCacheEntity)
                    .filter(UrlCacheEntity.url_hash == url_hash)
                    .delete()
                )
                return deleted > 0
        except Exception as e:
            logger.error(f"UrlCacheRepository: Error deleting hash '{url_hash}': {e}", exc_info=True)
            raise


url_cache_repository = UrlCacheRepository()
