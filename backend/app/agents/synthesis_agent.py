"""
Synthesis Agent — Context and Memory Merging.

Merges verified document chunks, web results, and user memories into a grounded
prompt with proper system instructions for final LLM generation.
"""

import time
from typing import Any

from backend.app.agents.protocols import AgentResult, AgentRole, AgentTrace
from backend.app.core.logging import logger


class SynthesisAgent:
    """Synthesis agent that assembles grounded prompts from verified context sources."""

    def __init__(self):
        """Initializes SynthesisAgent with lazy service references."""
        self._rag_service = None
        self._memory_service = None

    @property
    def rag(self):
        """Lazily imports and returns the rag_service singleton."""
        if self._rag_service is None:
            from backend.app.services.rag_service import rag_service
            self._rag_service = rag_service
        return self._rag_service

    @property
    def memory(self):
        """Lazily imports and returns the memory_service singleton."""
        if self._memory_service is None:
            from backend.app.services.memory_service import memory_service
            self._memory_service = memory_service
        return self._memory_service

    async def synthesize_context_async(
        self,
        query: str,
        doc_chunks: list[dict[str, Any]],
        web_chunks: list[dict[str, Any]],
        chat_history: list[dict[str, str]],
        vector_store: Any = None,
        user_id: str | None = None,
        user_memories: list[str] | None = None,
        document_metadata: dict[str, Any] | None = None,
        crag_relevant: bool = True,
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Merges document chunks, web results, and memories into a grounded prompt.

        Returns AgentResult with output keys: prompt, system_instruction, context_chunks, exported_chunks.
        """
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_synthesize"
        has_doc = bool(vector_store)
        doc_meta = dict(document_metadata or {})

        # Resolve document metadata
        if not doc_meta.get("title"):
            if vector_store and hasattr(vector_store, "metadata") and vector_store.metadata and vector_store.metadata.get("title"):
                doc_meta.update(vector_store.metadata)
            elif vector_store and hasattr(vector_store, "documents") and vector_store.documents:
                doc_meta["title"] = vector_store.documents[0].metadata.get("title", "")
                doc_meta["url"] = vector_store.documents[0].metadata.get("url", "")
            elif doc_chunks:
                doc_meta["title"] = doc_chunks[0].get("title", "")
                doc_meta["url"] = doc_chunks[0].get("url", "")

        # Document isolation: when a document is active, prevent web context bleeding
        if has_doc:
            effective_web_chunks: list[dict[str, Any]] = []
            # If no doc chunks and document is active, supply introductory chunks
            if not doc_chunks and vector_store and hasattr(vector_store, "documents") and vector_store.documents:
                sorted_docs = sorted(
                    vector_store.documents,
                    key=lambda d: d.metadata.get("chunk_index", 0),
                )
                for doc in sorted_docs[:3]:
                    doc_chunks.append({
                        "content": doc.metadata.get("parent_text") or doc.page_content,
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "parent_id": doc.metadata.get("parent_id", ""),
                        "url": doc.metadata.get("url", ""),
                        "title": doc.metadata.get("title", ""),
                        "rerank_score": 0.5,
                    })
        else:
            effective_web_chunks = list(web_chunks)

        combined_chunks = doc_chunks + effective_web_chunks

        # Retrieve semantic user memories
        effective_memories = list(user_memories or [])
        if not effective_memories and user_id:
            try:
                effective_memories = self.memory.get_relevant_memories(
                    query=query, user_id=user_id, limit=3
                )
            except Exception as mem_e:
                logger.debug(f"SynthesisAgent memory retrieval note: {mem_e}")

        # Format the grounded prompt
        prompt, system_instruction = self.rag.format_prompt(
            query=query,
            context_chunks=combined_chunks,
            chat_history=chat_history,
            user_memories=effective_memories if effective_memories else None,
            document_metadata=doc_meta if doc_meta else None,
        )

        # Suppress citations for out-of-context queries
        if has_doc and not crag_relevant:
            exported_chunks: list[dict[str, Any]] = []
        else:
            exported_chunks = combined_chunks

        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"SynthesisAgent: Synthesized prompt with {len(combined_chunks)} sources, "
            f"{len(effective_memories)} memories ({duration_ms:.1f}ms)"
        )

        if trace:
            trace.add_step(
                agent_role=AgentRole.SYNTHESIZER,
                action="synthesize_context",
                input_summary=f"{len(doc_chunks)} docs, {len(effective_web_chunks)} web, {len(effective_memories)} memories",
                output_summary=f"Prompt ready, {len(exported_chunks)} exported chunks",
                duration_ms=duration_ms,
            )

        return AgentResult(
            task_id=task_id,
            agent_role=AgentRole.SYNTHESIZER,
            output={
                "prompt": prompt,
                "system_instruction": system_instruction,
                "context_chunks": combined_chunks,
                "exported_chunks": exported_chunks,
            },
            execution_time_ms=duration_ms,
        )


synthesis_agent = SynthesisAgent()
