"""
Supervisor Agent — Multi-Agent Orchestrator.

Manages the multi-agent workflow by delegating tasks to specialized agents,
tracking agent handoffs, aggregating telemetry, and producing execution traces.
"""

import time
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from backend.app.agents.critic_agent import critic_agent
from backend.app.agents.planner_agent import planner_agent
from backend.app.agents.protocols import AgentTrace
from backend.app.agents.research_agent import research_agent
from backend.app.agents.router_agent import router_agent
from backend.app.agents.synthesis_agent import synthesis_agent
from backend.app.core.logging import logger
from backend.app.dtos.chat_dto import ChatMessageDto


class SupervisorWorkflowResult(BaseModel):
    """Structured result returned by the multi-agent workflow."""
    answer: str
    route: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    model_used: str = "gemini-2.0-flash"
    model_used: str = "gemini-3.6-flash"
    provider: str = "gemini"
    fallback_triggered: bool = False
    latency_sec: float = 0.0
    trace: AgentTrace | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)


class SupervisorAgent:
    """Orchestrator agent that manages the multi-agent RAG workflow."""

    def __init__(self):
        """Initializes SupervisorAgent with references to all specialized agents."""
        self.router = router_agent
        self.planner = planner_agent
        self.researcher = research_agent
        self.critic = critic_agent
        self.synthesizer = synthesis_agent
        self._llm_service = None

    @property
    def llm(self):
        """Lazily imports and returns the llm_service singleton."""
        if self._llm_service is None:
            from backend.app.services.llm_service import llm_service
            self._llm_service = llm_service
        return self._llm_service

    async def run_workflow_async(
        self,
        query: str,
        vector_store: Any = None,
        url: str | None = None,
        document_content: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
        selected_model: dict[str, str] | None = None,
        user_id: str | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> SupervisorWorkflowResult:
        """Runs end-to-end multi-agent workflow with generation, Self-RAG reflection, and tracing."""
        start_time = time.time()

        # Step 1: Orchestrate multi-agent retrieval and synthesis
        history_dtos = [ChatMessageDto(**m) if isinstance(m, dict) else m for m in (chat_history or [])]
        context_chunks, prompt, system_instruction, telemetry, trace = await self.orchestrate_async(
            query=query,
            vector_store=vector_store,
            chat_history=history_dtos,
            selected_model=selected_model,
            user_id=user_id,
            document_metadata=document_metadata,
        )

        # Step 2: Generate answer
        gen_res = await self.llm.generate_response_async(
            prompt=prompt,
            system_instruction=system_instruction,
            selected_model=selected_model,
        )
        answer = gen_res.get("text", "")

        # Step 3: Self-RAG Groundedness & Faithfulness Reflection
        if context_chunks and len(answer) > 30:
            selfrag_result = await self.critic.grade_hallucination_async(
                query=query,
                answer=answer,
                context_chunks=context_chunks,
                trace=trace,
            )
            grounded = selfrag_result.output.get("grounded", True)
            useful = selfrag_result.output.get("useful", True)
            critique = selfrag_result.output.get("critique")

            telemetry["self_rag_evaluated"] = True
            telemetry["self_rag_grounded"] = grounded
            telemetry["self_rag_useful"] = useful

            if (not grounded or not useful) and critique:
                logger.info(f"SupervisorAgent: Self-RAG reflection triggered retry with critique: {critique}")
                revised_prompt = prompt + f"\n\n[CRITIQUE FROM PREVIOUS DRAFT]:\n{critique}\nPlease strictly fix these issues and remain completely grounded in the verified context."
                retry_res = await self.llm.generate_response_async(
                    prompt=revised_prompt,
                    system_instruction=system_instruction,
                    selected_model=selected_model,
                )
                answer = retry_res.get("text", answer)

        total_latency = time.time() - start_time
        trace.total_duration_ms = total_latency * 1000

        return SupervisorWorkflowResult(
            answer=answer,
            route=telemetry.get("route", "DOCUMENT_RAG"),
            citations=context_chunks,
            model_used=gen_res.get("model_used", "gemini-2.0-flash"),
            provider=gen_res.get("provider", "gemini"),
            fallback_triggered=gen_res.get("fallback_triggered", False),
            latency_sec=total_latency,
            trace=trace,
            telemetry=telemetry,
        )

    async def orchestrate_async(
        self,
        query: str,
        vector_store: Any = None,
        chat_history: list[ChatMessageDto] | None = None,
        selected_model: dict[str, str] | None = None,
        user_id: str | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str, str, dict[str, Any], AgentTrace]:
        """Executes the full multi-agent RAG pipeline.

        Returns: (context_chunks, prompt, system_instruction, telemetry, trace)
        """
        trace = AgentTrace(query=query)
        history_dicts = []
        if chat_history:
            history_dicts = [{"role": m.role, "content": m.content} for m in chat_history]

        has_doc = bool(vector_store)
        doc_meta = self._resolve_metadata(vector_store, document_metadata)
        doc_title = doc_meta.get("title", "")

        # ---- Step 1: Router Agent — Intent Classification ----
        route_result = await self.router.classify_intent_async(
            query=query, has_document=has_doc, document_title=doc_title, trace=trace,
        )
        route = route_result.output.get("route", "DOCUMENT_RAG")
        route_confidence = route_result.output.get("confidence", 0.8)
        route_method = route_result.output.get("method", "unknown")

        # ---- Step 2: Branch based on route ----
        doc_chunks: list[dict[str, Any]] = []
        web_chunks: list[dict[str, Any]] = []
        crag_relevant = True
        crag_confidence = 1.0
        rewrite_count = 0

        if route == "DIRECT_CHAT":
            # Skip retrieval entirely
            pass

        elif route == "WEB_SEARCH":
            web_result = await self.researcher.web_search_async(query=query, trace=trace)
            web_chunks = web_result.output.get("web_results", [])

        else:
            # DOCUMENT_RAG or COMPLEX_ANALYTIC — full retrieval pipeline
            # Step 2a: Planner — decompose query
            plan_result = await self.planner.decompose_query_async(query=query, trace=trace)
            sub_queries = plan_result.output.get("sub_queries", [query])

            # Step 2b: Research — hybrid retrieval
            retrieve_result = await self.researcher.retrieve_documents_async(
                query=query, sub_queries=sub_queries,
                vector_store=vector_store, original_query=query, trace=trace,
            )
            doc_chunks = retrieve_result.output.get("documents", [])

            # Step 2c: Critic — CRAG document grading
            crag_result = await self.critic.grade_documents_async(
                query=query, documents=doc_chunks, original_query=query, trace=trace,
            )
            crag_relevant = crag_result.output.get("is_relevant", True)
            crag_confidence = crag_result.output.get("confidence", 1.0)
            doc_chunks = crag_result.output.get("filtered_documents", doc_chunks)

            # Step 2d: Iterative query rewrite loop if documents are not relevant
            if not crag_relevant and crag_confidence < 0.4 and rewrite_count < 1:
                rewrite_result = await self.planner.rewrite_query_async(
                    query=query,
                    critique=crag_result.output.get("reasoning"),
                    trace=trace,
                )
                rewritten = rewrite_result.output.get("rewritten_query", query)
                rewrite_count += 1

                # Re-retrieve with rewritten query
                re_retrieve = await self.researcher.retrieve_documents_async(
                    query=rewritten, sub_queries=[rewritten],
                    vector_store=vector_store, original_query=query, trace=trace,
                )
                doc_chunks = re_retrieve.output.get("documents", [])

                # Re-grade
                re_crag = await self.critic.grade_documents_async(
                    query=rewritten, documents=doc_chunks, original_query=query, trace=trace,
                )
                crag_relevant = re_crag.output.get("is_relevant", True)
                crag_confidence = re_crag.output.get("confidence", 1.0)
                doc_chunks = re_crag.output.get("filtered_documents", doc_chunks)

            # Web fallback if still not relevant and no active document
            if not crag_relevant and not has_doc:
                web_result = await self.researcher.web_search_async(query=query, trace=trace)
                web_chunks = web_result.output.get("web_results", [])

        # ---- Step 3: Synthesis Agent — merge context ----
        synth_result = await self.synthesizer.synthesize_context_async(
            query=query, doc_chunks=doc_chunks, web_chunks=web_chunks,
            chat_history=history_dicts, vector_store=vector_store,
            user_id=user_id, document_metadata=doc_meta,
            crag_relevant=crag_relevant, trace=trace,
        )
        prompt = synth_result.output.get("prompt", "")
        system_instruction = synth_result.output.get("system_instruction", "")
        exported_chunks = synth_result.output.get("exported_chunks", [])

        # ---- Build telemetry ----
        telemetry = {
            "route": route,
            "route_confidence": route_confidence,
            "route_method": route_method,
            "crag_graded": True,
            "crag_relevant": crag_relevant,
            "crag_confidence": crag_confidence,
            "rewrite_count": rewrite_count,
            "web_search_triggered": len(web_chunks) > 0,
            "documents_count": len(doc_chunks),
            "web_results_count": len(web_chunks),
            "agent_handoffs": trace.agent_handoffs,
            "self_rag_evaluated": False,
            "self_rag_grounded": True,
        }

        trace.finalize()
        return exported_chunks, prompt, system_instruction, telemetry, trace

    async def orchestrate_stream_async(
        self,
        query: str,
        vector_store: Any = None,
        chat_history: list[ChatMessageDto] | None = None,
        selected_model: dict[str, str] | None = None,
        user_id: str | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Executes the multi-agent RAG pipeline yielding step events for SSE streaming.

        Yields dicts with type: "step", "ready", etc.
        """
        trace = AgentTrace(query=query)
        history_dicts = []
        if chat_history:
            history_dicts = [{"role": m.role, "content": m.content} for m in chat_history]

        has_doc = bool(vector_store)
        doc_meta = self._resolve_metadata(vector_store, document_metadata)
        doc_title = doc_meta.get("title", "")

        # ---- Step 1: Router Agent ----
        route_result = await self.router.classify_intent_async(
            query=query, has_document=has_doc, document_title=doc_title, trace=trace,
        )
        route = route_result.output.get("route", "DOCUMENT_RAG")
        route_method = route_result.output.get("method", "unknown")
        detail_map = {
            "DOCUMENT_RAG": "Routing query to Document Knowledge Base",
            "WEB_SEARCH": "Routing query to Autonomous Web Search Engine",
            "DIRECT_CHAT": "Direct Conversational Reasoning",
            "COMPLEX_ANALYTIC": "Complex Analytic Pipeline (multi-hop retrieval)",
        }
        yield {
            "type": "step", "step": "route", "icon": "•",
            "title": f"AI Intent Classification ({route_method})",
            "detail": detail_map.get(route, f"Classified: {route}"),
            "status": "done",
        }

        doc_chunks: list[dict[str, Any]] = []
        web_chunks: list[dict[str, Any]] = []
        crag_relevant = True
        crag_confidence = 1.0
        rewrite_count = 0

        if route == "DIRECT_CHAT":
            pass
        elif route == "WEB_SEARCH":
            web_result = await self.researcher.web_search_async(query=query, trace=trace)
            web_chunks = web_result.output.get("web_results", [])
            yield {
                "type": "step", "step": "web_search", "icon": "•",
                "title": "Autonomous Web Fallback",
                "detail": f"Retrieved {len(web_chunks)} external references via DuckDuckGo & Wikipedia API",
                "status": "done",
            }
        else:
            # Planner
            plan_result = await self.planner.decompose_query_async(query=query, trace=trace)
            sub_queries = plan_result.output.get("sub_queries", [query])
            if len(sub_queries) > 1:
                yield {
                    "type": "step", "step": "decompose", "icon": "•",
                    "title": "Sub-Question Decomposition",
                    "detail": f"Deconstructed into {len(sub_queries)} focused sub-queries for multi-hop retrieval",
                    "sub_queries": sub_queries, "status": "done",
                }
            else:
                yield {
                    "type": "step", "step": "decompose", "icon": "•",
                    "title": "Query Semantic Expansion",
                    "detail": "Generated HyDE embeddings & lexical keyword variations",
                    "status": "done",
                }

            # Research
            retrieve_result = await self.researcher.retrieve_documents_async(
                query=query, sub_queries=sub_queries,
                vector_store=vector_store, original_query=query, trace=trace,
            )
            doc_chunks = retrieve_result.output.get("documents", [])
            yield {
                "type": "step", "step": "retrieve", "icon": "•",
                "title": "Hierarchical Hybrid Retrieval",
                "detail": f"Retrieved {len(doc_chunks)} context chunks via Qdrant BQ + FAISS BM25 + FlashRank Reranker",
                "status": "done",
            }

            # CRAG Grading
            crag_result = await self.critic.grade_documents_async(
                query=query, documents=doc_chunks, original_query=query, trace=trace,
            )
            crag_relevant = crag_result.output.get("is_relevant", True)
            crag_confidence = crag_result.output.get("confidence", 1.0)
            doc_chunks = crag_result.output.get("filtered_documents", doc_chunks)
            pct = int(crag_confidence * 100) if crag_confidence <= 1.0 else int(crag_confidence)
            crag_detail = (
                f"Relevance confidence: {pct}% (Context verified & grounded)"
                if crag_relevant
                else f"Relevance confidence: {pct}% (Insufficient context, initiating re-routing)"
            )
            yield {
                "type": "step", "step": "crag", "icon": "•",
                "title": "CRAG Document Relevance Grader",
                "detail": crag_detail, "status": "done",
            }

            # Query rewrite loop
            if not crag_relevant and crag_confidence < 0.4 and rewrite_count < 1:
                rewrite_result = await self.planner.rewrite_query_async(
                    query=query, critique=crag_result.output.get("reasoning"), trace=trace,
                )
                rewritten = rewrite_result.output.get("rewritten_query", query)
                rewrite_count += 1
                yield {
                    "type": "step", "step": "transform_query", "icon": "•",
                    "title": "Iterative Query Refinement",
                    "detail": f"Refined search query to: '{rewritten}'",
                    "status": "done",
                }

                re_retrieve = await self.researcher.retrieve_documents_async(
                    query=rewritten, sub_queries=[rewritten],
                    vector_store=vector_store, original_query=query, trace=trace,
                )
                doc_chunks = re_retrieve.output.get("documents", [])
                yield {
                    "type": "step", "step": "retrieve", "icon": "•",
                    "title": "Re-Retrieval After Refinement",
                    "detail": f"Retrieved {len(doc_chunks)} chunks with refined query",
                    "status": "done",
                }

                re_crag = await self.critic.grade_documents_async(
                    query=rewritten, documents=doc_chunks, original_query=query, trace=trace,
                )
                crag_relevant = re_crag.output.get("is_relevant", True)
                crag_confidence = re_crag.output.get("confidence", 1.0)
                doc_chunks = re_crag.output.get("filtered_documents", doc_chunks)

            if not crag_relevant and not has_doc:
                web_result = await self.researcher.web_search_async(query=query, trace=trace)
                web_chunks = web_result.output.get("web_results", [])
                yield {
                    "type": "step", "step": "web_search", "icon": "•",
                    "title": "Autonomous Web Fallback",
                    "detail": f"Retrieved {len(web_chunks)} external references via DuckDuckGo & Wikipedia API",
                    "status": "done",
                }

        # Synthesis
        synth_result = await self.synthesizer.synthesize_context_async(
            query=query, doc_chunks=doc_chunks, web_chunks=web_chunks,
            chat_history=history_dicts, vector_store=vector_store,
            user_id=user_id, document_metadata=doc_meta,
            crag_relevant=crag_relevant, trace=trace,
        )
        prompt = synth_result.output.get("prompt", "")
        system_instruction = synth_result.output.get("system_instruction", "")
        exported_chunks = synth_result.output.get("exported_chunks", [])
        yield {
            "type": "step", "step": "synthesize", "icon": "•",
            "title": "Context & Memory Synthesis",
            "detail": f"Synthesized grounded prompt with {len(exported_chunks)} sources and user memory",
            "status": "done",
        }

        # ---- Skip Self-RAG Reflection in Streaming Path ----
        # In streaming mode, we skip generating a full provisional answer and hallucination grading
        # to avoid double-generation and reduce time-to-first-token latency.

        # Build telemetry
        telemetry = {
            "route": route,
            "route_method": route_method,
            "crag_relevant": crag_relevant,
            "crag_confidence": crag_confidence,
            "rewrite_count": rewrite_count,
            "web_search_triggered": len(web_chunks) > 0,
            "documents_count": len(doc_chunks),
            "web_results_count": len(web_chunks),
            "agent_handoffs": trace.agent_handoffs,
            "self_rag_evaluated": False,
            "self_rag_grounded": True,
        }

        trace.finalize()

        yield {
            "type": "ready",
            "context_chunks": exported_chunks,
            "prompt": prompt,
            "system_instruction": system_instruction,
            "telemetry": telemetry,
            "trace": trace.model_dump(),
        }

    @staticmethod
    def _resolve_metadata(
        vector_store: Any,
        document_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Resolves document metadata from vector store or provided metadata."""
        doc_meta = dict(document_metadata or {})
        if not doc_meta and vector_store:
            if hasattr(vector_store, "metadata") and vector_store.metadata:
                doc_meta = dict(vector_store.metadata)
            elif hasattr(vector_store, "documents") and vector_store.documents:
                doc_meta = {
                    "title": vector_store.documents[0].metadata.get("title", ""),
                    "url": vector_store.documents[0].metadata.get("url", ""),
                }
        return doc_meta


supervisor_agent = SupervisorAgent()
