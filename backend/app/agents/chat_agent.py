import asyncio
import queue
import threading
from collections.abc import AsyncGenerator, Generator
from typing import Any

from backend.app.agents.supervisor_agent import supervisor_agent
from backend.app.cache.vector_cache import vector_store_cache
from backend.app.core.errors import ValidationException
from backend.app.core.logging import logger
from backend.app.dtos.chat_dto import ChatMessageDto, ChatResponseDto, CitationItemDto
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.services.llm_service import llm_service
from backend.app.services.memory_service import memory_service
from backend.app.services.rag_service import rag_service


class WebChatAgent:
    """Autonomous conversational agent orchestrating multi-agent collaboration and generation."""

    def __init__(self):
        """Initializes chat agent with RAG, LLM, cache, memory, and supervisor agent."""
        try:
            self.rag = rag_service
            self.llm = llm_service
            self.vector_cache = vector_store_cache
            self.memory = memory_service
            self.supervisor = supervisor_agent
        except Exception as e:
            logger.error(f"Error initializing WebChatAgent: {e}", exc_info=True)
            raise

    def _resolve_vector_store(
        self,
        vector_store: Any | None,
        url: str | None,
        document_content: str | None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """Resolves vector store and metadata from parameter, RAM cache, or persistent database cache."""
        vs = vector_store
        meta = document_metadata
        if not vs and url:
            url_hash = compute_url_hash(url)
            vs = self.vector_cache.get(url) or self.vector_cache.get(url_hash)
            meta = meta or self.vector_cache.get_metadata(url) or self.vector_cache.get_metadata(url_hash)
            if not vs:
                try:
                    from backend.app.repositories.url_cache_repository import url_cache_repository
                    cache_entry = url_cache_repository.get_by_hash(url_hash)
                    if cache_entry:
                        meta = meta or cache_entry.get("metadata") or {
                            "url": url,
                            "title": cache_entry.get("title", ""),
                            "images": cache_entry.get("images", []),
                        }
                        vec_sid = cache_entry.get("vector_session_id")
                        if vec_sid:
                            vs = self.rag.get_session_index(session_id=vec_sid, metadata=meta)
                            if vs:
                                self.vector_cache.set(url_hash, vs, metadata=meta)
                except Exception as hydrate_err:
                    logger.warning(f"WebChatAgent: Could not re-hydrate vector store from persistent cache: {hydrate_err}")

        if not vs and document_content:
            try:
                vs, err = self.rag.build_vectorstore(
                    document_content, metadata=meta or {"url": url or "", "title": "Provided Document"}
                )
                if not err and vs and url:
                    self.vector_cache.set(compute_url_hash(url), vs, metadata=meta)
            except Exception as build_err:
                logger.warning(f"WebChatAgent: Failed to build on-the-fly vector store: {build_err}")

        return vs, meta

    async def answer_query_async(
        self,
        query: str,
        url: str | None = None,
        document_content: str | None = None,
        chat_history: list[ChatMessageDto] | None = None,
        selected_model: dict[str, str] | None = None,
        user_id: str | None = None,
        vector_store: Any | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> ChatResponseDto:
        """Executes full Multi-Agent RAG workflow with Self-RAG reflection asynchronously."""
        try:
            if not query or not query.strip():
                raise ValidationException("Query parameter cannot be empty.")

            vs, document_metadata = self._resolve_vector_store(vector_store, url, document_content, document_metadata)

            # Run supervisor multi-agent workflow
            result = await self.supervisor.run_workflow_async(
                query=query,
                vector_store=vs,
                url=url,
                document_content=document_content,
                chat_history=[{"role": m.role, "content": m.content} for m in chat_history] if chat_history else [],
                selected_model=selected_model,
                user_id=user_id,
                document_metadata=document_metadata,
            )

            citations = [
                CitationItemDto(
                    chunk_index=c.get("chunk_index", 0),
                    content=c.get("content", "")[:300] + "..." if len(c.get("content", "")) > 300 else c.get("content", ""),
                    title=c.get("title", ""),
                    url=c.get("url", ""),
                )
                for c in result.citations
            ]

            trace_dict = result.trace.model_dump() if result.trace else None

            return ChatResponseDto(
                answer=result.answer,
                model_used=result.model_used,
                provider=result.provider,
                fallback_triggered=result.fallback_triggered,
                latency_sec=result.latency_sec,
                citations=citations,
                agent_route=result.route,
                agent_trace=trace_dict,
            )
        except Exception as e:
            logger.error(f"Error executing answer_query_async for query='{query}': {e}", exc_info=True)
            raise

    async def answer_query_stream_async(
        self,
        query: str,
        url: str | None = None,
        document_content: str | None = None,
        chat_history: list[ChatMessageDto] | None = None,
        selected_model: dict[str, str] | None = None,
        user_id: str | None = None,
        vector_store: Any | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], AsyncGenerator[dict[str, Any], None]]:
        """Prepares context via supervisor and returns citations with token generator for SSE streaming."""
        try:
            vs, context_chunks, prompt, system_instruction, telemetry = await self._prepare_context_async(
                query=query,
                url=url,
                document_content=document_content,
                chat_history=chat_history,
                user_id=user_id,
                vector_store=vector_store,
                document_metadata=document_metadata,
            )

            citations_data = [
                {
                    "chunk_index": c.get("chunk_index", 0),
                    "content": c.get("content", "")[:250] + "...",
                    "title": c.get("title", ""),
                    "url": c.get("url", ""),
                }
                for c in context_chunks
            ]

            stream_gen = self.llm.generate_stream_async(
                prompt=prompt,
                system_instruction=system_instruction,
                selected_model=selected_model,
            )

            return citations_data, stream_gen
        except Exception as e:
            logger.error(f"Error executing answer_query_stream_async: {e}", exc_info=True)
            raise

    async def answer_query_stream_events_async(
        self,
        query: str,
        url: str | None = None,
        document_content: str | None = None,
        chat_history: list[ChatMessageDto] | None = None,
        selected_model: dict[str, str] | None = None,
        user_id: str | None = None,
        vector_store: Any | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Executes multi-agent pipeline yielding step events, citations, and LLM token stream asynchronously."""
        try:
            if not query or not query.strip():
                raise ValidationException("Query parameter cannot be empty.")

            vs, document_metadata = self._resolve_vector_store(vector_store, url, document_content, document_metadata)

            # Stream Supervisor multi-agent reasoning steps (including Self-RAG reflection)
            final_prompt = ""
            final_system_instruction = ""
            context_chunks: list[dict[str, Any]] = []

            async for item in self.supervisor.orchestrate_stream_async(
                query=query,
                vector_store=vs,
                chat_history=chat_history,
                selected_model=selected_model,
                user_id=user_id,
                document_metadata=document_metadata,
            ):
                if item.get("type") == "step":
                    yield item
                elif item.get("type") == "ready":
                    context_chunks = item.get("context_chunks", [])
                    final_prompt = item.get("prompt", "")
                    final_system_instruction = item.get("system_instruction", "")

            citations_data = [
                {
                    "chunk_index": c.get("chunk_index", 0),
                    "content": c.get("content", "")[:250] + "...",
                    "title": c.get("title", ""),
                    "url": c.get("url", ""),
                }
                for c in context_chunks
            ]
            yield {"type": "citations", "citations": citations_data}

            # Stream LLM tokens
            async for chunk in self.llm.generate_stream_async(
                prompt=final_prompt,
                system_instruction=final_system_instruction,
                selected_model=selected_model,
            ):
                yield {"type": "chunk", **chunk}
        except Exception as e:
            logger.error(f"Error in answer_query_stream_events_async: {e}", exc_info=True)
            raise

    def answer_query_stream_events(
        self,
        query: str,
        url: str | None = None,
        document_content: str | None = None,
        chat_history: list[ChatMessageDto] | None = None,
        selected_model: dict[str, str] | None = None,
        user_id: str | None = None,
        vector_store: Any | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Synchronous generator wrapper around answer_query_stream_events_async for Streamlit UI."""
        try:
            q: queue.Queue = queue.Queue()
            sentinel = object()

            async def _producer():
                try:
                    async for event in self.answer_query_stream_events_async(
                        query=query,
                        url=url,
                        document_content=document_content,
                        chat_history=chat_history,
                        selected_model=selected_model,
                        user_id=user_id,
                        vector_store=vector_store,
                        document_metadata=document_metadata,
                    ):
                        q.put(event)
                except Exception as exc:
                    q.put(exc)
                finally:
                    q.put(sentinel)

            def _worker():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_producer())
                finally:
                    loop.close()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            while True:
                item = q.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except Exception as e:
            logger.error(f"Error in answer_query_stream_events: {e}", exc_info=True)
            raise

    async def _prepare_context_async(
        self,
        query: str,
        url: str | None = None,
        document_content: str | None = None,
        chat_history: list[ChatMessageDto] | None = None,
        user_id: str | None = None,
        vector_store: Any | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], str, str, dict[str, Any]]:
        """Executes context preparation via supervisor agent orchestration."""
        try:
            if not query or not query.strip():
                raise ValidationException("Query parameter cannot be empty.")

            vs, document_metadata = self._resolve_vector_store(vector_store, url, document_content, document_metadata)

            # Orchestrate via Supervisor
            context_chunks, prompt, system_instruction, telemetry, trace = await self.supervisor.orchestrate_async(
                query=query,
                vector_store=vs,
                chat_history=chat_history,
                user_id=user_id,
                document_metadata=document_metadata,
            )

            return vs, context_chunks, prompt, system_instruction, telemetry
        except Exception as e:
            logger.error(f"Error in _prepare_context_async: {e}", exc_info=True)
            raise


chat_agent = WebChatAgent()
