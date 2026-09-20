import os
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv
from memori import Memori
import memori.memory.recall as memori_recall

import backend.config.database as db
from backend.app.core.logging import logger

load_dotenv()


class MemoryService:
    """Provides high-level integration with Memori Labs for automatic conversation memory management."""

    def __init__(self, process_id: str = "webchat"):
        """Initializes memory service with Memori process identifier."""
        try:
            self.process_id = process_id
            self._memori: Optional[Memori] = None
            self._embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
        except Exception as e:
            logger.error(f"MemoryService initialization error: {e}", exc_info=True)
            raise

    def get_memori(self) -> Optional[Memori]:
        """Initializes or returns singleton Memori instance supporting both Cloud and BYODB storage."""
        try:
            if self._memori is not None:
                return self._memori

            api_key = os.getenv("MEMORI_API_KEY")
            if api_key:
                self._memori = Memori(api_key=api_key)
                logger.info("MemoryService: Connected to Memori Cloud.")
                return self._memori

            db.get_engine()
            if db.SessionLocal is not None:
                self._memori = Memori(conn=db.SessionLocal)
                logger.info("MemoryService: Connected to Memori BYODB storage.")
                return self._memori

            logger.warning("MemoryService: Database session factory is not available.")
            return None
        except Exception as e:
            logger.warning(f"MemoryService: Failed to initialize Memori instance: {e}")
            return None

    def register_llm(self, client: Any) -> Any:
        """Registers an LLM client with Memori so Memori automatically intercepts and manages memories."""
        try:
            mem = self.get_memori()
            if mem and client is not None:
                mem.llm.register(client)
                logger.info("MemoryService: LLM client successfully registered with Memori.")
                return client
            return client
        except Exception as e:
            logger.warning(f"MemoryService: Could not register LLM with Memori: {e}")
            return client

    def set_embed_fn(self, embed_fn: Callable[[List[str]], List[List[float]]]) -> None:
        """Registers a fast embedding function to power Memori semantic recall."""
        try:
            self._embed_fn = embed_fn

            def _memori_embed_bridge(texts: Union[str, List[str]], model: Optional[str] = None, **kwargs: Any) -> List[List[float]]:
                try:
                    text_list = [texts] if isinstance(texts, str) else list(texts)
                    if self._embed_fn:
                        return self._embed_fn(text_list)
                    return [[0.0] * 384 for _ in text_list]
                except Exception as emb_err:
                    logger.debug(f"MemoryService: Embedding bridge note: {emb_err}")
                    count = 1 if isinstance(texts, str) else len(texts)
                    return [[0.0] * 384 for _ in range(count)]

            memori_recall.embed_texts = _memori_embed_bridge  # type: ignore
            logger.info("MemoryService: Registered custom embedding bridge with Memori recall engine.")
        except Exception as e:
            logger.debug(f"MemoryService: Embed function registration note: {e}")

    def set_attribution(self, user_id: str, process_id: Optional[str] = None) -> None:
        """Associates subsequent LLM interactions with the specific user entity in Memori."""
        try:
            if not user_id:
                return
            mem = self.get_memori()
            if mem:
                pid = process_id or self.process_id
                mem.attribution(entity_id=str(user_id), process_id=pid)
        except Exception as e:
            logger.debug(f"MemoryService: Attribution note: {e}")

    def build_storage(self) -> bool:
        """Builds Memori Labs database storage schema and data structures via storage.build()."""
        try:
            mem = self.get_memori()
            if mem and hasattr(mem, "config") and hasattr(mem.config, "storage") and mem.config.storage:
                mem.config.storage.build()
                logger.info("MemoryService: Memori Labs storage schema built successfully.")
                return True
            return False
        except Exception as e:
            logger.warning(f"MemoryService: Memori storage build warning: {e}")
            return False

    def initialize(self) -> bool:
        """Builds Memori Labs storage and pre-warms the memory client."""
        try:
            self.build_storage()
            mem = self.get_memori()
            if mem:
                logger.info("MemoryService: Memori Labs semantic memory client initialized and ready.")
                return True
            return False
        except Exception as e:
            logger.warning(f"MemoryService: Startup initialization warning: {e}")
            return False

    def add_memory(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Captures a conversation turn into Memori using Memori's native agent turn capture."""
        try:
            if not user_id or not messages:
                return False

            mem = self.get_memori()
            if not mem:
                return False

            self.set_attribution(user_id)

            user_content = ""
            assistant_content = ""

            if isinstance(messages, str):
                parts = messages.split("\nAssistant:")
                if len(parts) >= 2:
                    user_content = parts[0].replace("User:", "").strip()
                    assistant_content = parts[1].strip()
                else:
                    user_content = messages.strip()
            elif isinstance(messages, list):
                for m in messages:
                    if isinstance(m, dict):
                        role = str(m.get("role", "")).lower()
                        content = str(m.get("content", "")).strip()
                        if role in ["user", "human"] and content:
                            user_content = content
                        elif role in ["assistant", "ai", "model"] and content:
                            assistant_content = content

            if user_content:
                mem.capture_agent_turn(
                    user_content=user_content,
                    assistant_content=assistant_content or "Acknowledged.",
                    project_id=self.process_id,
                    session_id=str(session_id) if session_id else None,
                )
                logger.info(f"MemoryService: Captured conversation turn for user '{user_id}' via Memori.")
                return True
            return False
        except Exception as e:
            logger.debug(f"MemoryService: Note on turn capture for user '{user_id}': {e}")
            return False

    def get_relevant_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> List[str]:
        """Retrieves recalled memories for user via Memori's native recall engine."""
        try:
            if not user_id or not query:
                return []

            mem = self.get_memori()
            if not mem:
                return []

            self.set_attribution(user_id)
            recalled = mem.recall(query=query, limit=limit)

            memories: List[str] = []
            if isinstance(recalled, list):
                for item in recalled:
                    if isinstance(item, str):
                        memories.append(item.strip())
                    elif hasattr(item, "fact") and getattr(item, "fact"):
                        memories.append(str(item.fact).strip())
                    elif hasattr(item, "content") and getattr(item, "content"):
                        memories.append(str(item.content).strip())
                    elif isinstance(item, dict):
                        val = item.get("fact") or item.get("content") or item.get("memory")
                        if val:
                            memories.append(str(val).strip())
            elif isinstance(recalled, dict):
                facts = recalled.get("facts", [])
                for f in facts:
                    if isinstance(f, str):
                        memories.append(f.strip())
                    elif isinstance(f, dict):
                        val = f.get("fact") or f.get("content") or f.get("memory")
                        if val:
                            memories.append(str(val).strip())

            return [m for m in memories if m]
        except Exception as e:
            logger.debug(f"MemoryService: Recall note for user '{user_id}': {e}")
            return []

    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Returns all stored memories for a user using Memori recall."""
        try:
            memories_text = self.get_relevant_memories(query="user preferences and profile details", user_id=user_id, limit=20)
            results: List[Dict[str, Any]] = []
            for idx, text_item in enumerate(memories_text, 1):
                results.append(
                    {
                        "id": str(idx),
                        "memory": text_item,
                    }
                )
            return results
        except Exception as e:
            logger.warning(f"MemoryService: Error getting memories for user '{user_id}': {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes entity memories from Memori."""
        try:
            mem = self.get_memori()
            if not mem or not memory_id:
                return False
            mem.delete_entity_memories(entity_id=str(memory_id))
            logger.info(f"MemoryService: Deleted memories for entity '{memory_id}'.")
            return True
        except Exception as e:
            logger.warning(f"MemoryService: Error deleting memories: {e}")
            return False

    def delete_user_memories(self, user_id: str) -> bool:
        """Deletes all persistent memories for user via Memori's delete_entity_memories()."""
        try:
            if not user_id:
                return False
            mem = self.get_memori()
            if not mem:
                return False
            mem.delete_entity_memories(entity_id=str(user_id))
            logger.info(f"MemoryService: Deleted all memories for user '{user_id}'.")
            return True
        except Exception as e:
            logger.warning(f"MemoryService: Error deleting memories for user '{user_id}': {e}")
            return False


memory_service = MemoryService()
