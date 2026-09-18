import asyncio
import queue
import threading
from collections.abc import AsyncGenerator, Generator
from typing import Any

from backend.app.agents.supervisor_agent import supervisor_agent
from backend.app.cache.vector_cache import vector_store_cache
from backend.app.core.errors import ValidationException, VectorStoreException
from backend.app.core.logging import logger
from backend.app.dtos.chat_dto import ChatMessageDto, ChatResponseDto, CitationItemDto
from backend.app.helpers.url_helper import compute_url_hash
from backend.app.services.agentic_rag_service import agentic_rag_service
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
            self.agentic_rag = agentic_rag_service
            self.supervisor = supervisor_agent
        except Exception as e:
            logger.error(f"Error initializing WebChatAgent: {e}", exc_info=True)
            raise

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

            vs = vector_store
            if not vs and url:
                vs = self.vector_cache.get(url) or self.vector_cache.get(compute_url_hash(url))

            # if not vs and document_content:
            #     vs, err = self.rag.build_vectorstore(
            #         document_content, metadata={"url": url or "", "title": "Provided Document"}
            #     )
            #     if err:
            #         raise VectorStoreException(f"Failed to build vector index: {err}")

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

            vs = vector_store
            if not vs and url:
                vs = self.vector_cache.get(url) or self.vector_cache.get(compute_url_hash(url))

            if not vs and document_content:
                vs, err = self.rag.build_vectorstore(
                    document_content, metadata={"url": url or "", "title": "Provided Document"}
                )
                if err:
                    raise VectorStoreException(f"Failed to build vector index: {err}")

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

            vs = vector_store
            if not vs and url:
                vs = self.vector_cache.get(url) or self.vector_cache.get(compute_url_hash(url))

            # if not vs and document_content:
            #     vs, err = self.rag.build_vectorstore(
            #         document_content, metadata={"url": url or "", "title": "Provided Document"}
            #     )
            #     if err:
            #         raise VectorStoreException(f"Failed to build vector index: {err}")

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
