"""
Research Agent — Hybrid Retrieval and Web Search.

Orchestrates document retrieval across Qdrant dense vectors, FAISS BM25 sparse keywords,
and FlashRank reranking, with autonomous DuckDuckGo and Wikipedia web search fallback.
"""

import asyncio
import re
import time
from typing import Any

import requests

from backend.app.agents.protocols import AgentResult, AgentRole, AgentTrace
from backend.app.core.logging import logger
from backend.app.services.agentic_rag_service import is_overview_or_summary_query
from backend.app.services.agentic_rag_service import (
    is_diagram_or_image_query,
    is_overview_or_summary_query,
)


class ResearchAgent:
    """Specialized retrieval agent for hybrid document search and web fallback tools."""

    def __init__(self):
        """Initializes ResearchAgent with lazy service references."""
        self._rag_service = None
        self._rerank_service = None

    @property
    def rag(self):
        """Lazily imports and returns the rag_service singleton."""
        if self._rag_service is None:
            from backend.app.services.rag_service import rag_service
            self._rag_service = rag_service
        return self._rag_service

    @property
    def reranker(self):
        """Lazily imports and returns the rerank_service singleton."""
        if self._rerank_service is None:
            from backend.app.services.rerank_service import rerank_service
            self._rerank_service = rerank_service
        return self._rerank_service

    async def retrieve_documents_async(
        self,
        query: str,
        sub_queries: list[str],
        vector_store: Any,
        original_query: str = "",
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Retrieves context chunks across sub-queries using hybrid search and reranking.

        Returns AgentResult with output keys: documents, total_chunks.
        """
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_retrieve"
        orig_q = original_query or query
        all_chunks: list[dict[str, Any]] = []
        seen_texts: set = set()

        # If it's an overview/summary query, prepend introductory document chunks
        if is_overview_or_summary_query(orig_q) and vector_store:
            if hasattr(vector_store, "documents") and vector_store.documents:
                sorted_docs = sorted(
                    vector_store.documents,
                    key=lambda d: d.metadata.get("chunk_index", 0),
                )
                for doc in sorted_docs[:4]:
                    content = doc.metadata.get("parent_text") or doc.page_content
                    text_hash = content[:100]
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        all_chunks.append({
                            "content": content,
                            "chunk_index": doc.metadata.get("chunk_index", 0),
                            "parent_id": doc.metadata.get("parent_id", ""),
                            "url": doc.metadata.get("url", ""),
                            "title": doc.metadata.get("title", ""),
                            "rerank_score": 1.0,
                        })

        # If it's a diagram/image query, inject all visual image diagram chunks
        if is_diagram_or_image_query(orig_q) and vector_store:
            if hasattr(vector_store, "documents") and vector_store.documents:
                for doc in vector_store.documents:
                    if doc.metadata.get("is_image") or "[Visual Content Image Diagram" in doc.page_content:
                        content = doc.metadata.get("parent_text") or doc.page_content
                        text_hash = content[:100]
                        if text_hash not in seen_texts:
                            seen_texts.add(text_hash)
                            all_chunks.append({
                                "content": content,
                                "chunk_index": doc.metadata.get("chunk_index", 0),
                                "parent_id": doc.metadata.get("parent_id", ""),
                                "url": doc.metadata.get("url", ""),
                                "title": doc.metadata.get("title", ""),
                                "rerank_score": 1.0,
                            })

            meta = getattr(vector_store, "metadata", None) or {}
            images = meta.get("images") or []
            for img in images:
                img_url = img.get("url")
                if img_url:
                    img_doc_content = (
                        f"[Visual Content Image Diagram / Illustration]\n"
                        f"Diagram Title: {img.get('alt', 'Architecture Diagram')}\n"
                        f"Image URL: {img_url}\n"
                        f"Caption & Architectural Context: {img.get('context') or img.get('alt')}"
                    )
                    text_hash = img_doc_content[:100]
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        all_chunks.append({
                            "content": img_doc_content,
                            "chunk_index": len(all_chunks),
                            "parent_id": f"img_direct_{len(all_chunks)}",
                            "url": meta.get("url", ""),
                            "title": meta.get("title", ""),
                            "rerank_score": 1.0,
                        })

        effective_sub_queries = list(sub_queries)
        if is_diagram_or_image_query(orig_q) and not any("diagram" in sq.lower() for sq in effective_sub_queries):
            effective_sub_queries.append("visual content image diagrams and architectural illustrations")

        # Retrieve across sub-queries using hybrid search in parallel
        tasks = [
            asyncio.to_thread(
                self.rag.retrieve_context,
                index_or_store=vector_store,
                query=sq,
                top_k=3,
                use_expansion=True,
                use_rerank=True,
            )
            for sq in effective_sub_queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for sq, result in zip(effective_sub_queries, results):
            if isinstance(result, Exception):
                logger.warning(f"ResearchAgent: Retrieval error for sub-query '{sq[:40]}': {result}")
            else:
                for c in result:
                    text_hash = c.get("content", "")[:100]
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        all_chunks.append(c)

        duration_ms = (time.time() - start) * 1000
        logger.info(f"ResearchAgent: Retrieved {len(all_chunks)} unique chunks across {len(sub_queries)} sub-queries ({duration_ms:.1f}ms)")

        if trace:
            trace.add_step(
                agent_role=AgentRole.RESEARCHER,
                action="hybrid_retrieve",
                input_summary=f"{len(sub_queries)} sub-queries",
                output_summary=f"{len(all_chunks)} chunks retrieved",
                duration_ms=duration_ms,
            )

        return AgentResult(
            task_id=task_id,
            agent_role=AgentRole.RESEARCHER,
            output={"documents": all_chunks, "total_chunks": len(all_chunks)},
            execution_time_ms=duration_ms,
        )

    async def web_search_async(
        self,
        query: str,
        max_results: int = 4,
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Executes autonomous web search via DuckDuckGo and Wikipedia APIs.

        Returns AgentResult with output keys: web_results, sources_count.
        """
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_web_search"

        web_results = await asyncio.to_thread(self._perform_web_search, query, max_results)

        duration_ms = (time.time() - start) * 1000
        logger.info(f"ResearchAgent: Web search returned {len(web_results)} results ({duration_ms:.1f}ms)")

        if trace:
            trace.add_step(
                agent_role=AgentRole.RESEARCHER,
                action="web_search",
                input_summary=query[:60],
                output_summary=f"{len(web_results)} web results",
                duration_ms=duration_ms,
            )

        return AgentResult(
            task_id=task_id,
            agent_role=AgentRole.RESEARCHER,
            output={"web_results": web_results, "sources_count": len(web_results)},
            execution_time_ms=duration_ms,
        )

    def _perform_web_search(self, query: str, max_results: int = 4) -> list[dict[str, Any]]:
        """Queries DuckDuckGo Instant Answer API and Wikipedia Search API."""
        web_results: list[dict[str, Any]] = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WebChat/2.0"}

        # DuckDuckGo Instant Knowledge API
        try:
            ddg_resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                headers=headers,
                timeout=5,
            )
            if ddg_resp.status_code == 200:
                data = ddg_resp.json()
                abstract = data.get("AbstractText", "")
                abstract_url = data.get("AbstractURL", "")
                heading = data.get("Heading", "DuckDuckGo Knowledge")

                if abstract:
                    web_results.append({
                        "content": f"[Web Knowledge: {heading}]\n{abstract}\nSource: {abstract_url}",
                        "chunk_index": 1001,
                        "title": heading,
                        "url": abstract_url or "https://duckduckgo.com",
                        "source_type": "web_search",
                    })

                for idx, item in enumerate(data.get("RelatedTopics", [])[:2]):
                    if isinstance(item, dict) and item.get("Text"):
                        web_results.append({
                            "content": f"[Web Topic: {heading}]\n{item.get('Text')}\nSource: {item.get('FirstURL', '')}",
                            "chunk_index": 1002 + idx,
                            "title": f"{heading} - Related",
                            "url": item.get("FirstURL", "https://duckduckgo.com"),
                            "source_type": "web_search",
                        })
        except Exception as e:
            logger.debug(f"ResearchAgent DuckDuckGo error: {e}")

        # Wikipedia Search API
        if len(web_results) < max_results:
            try:
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": max_results,
                }
                wiki_resp = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params=params,
                    headers=headers,
                    timeout=5,
                )
                if wiki_resp.status_code == 200:
                    search_items = wiki_resp.json().get("query", {}).get("search", [])
                    for i, item in enumerate(search_items[:max_results]):
                        title = item.get("title", "")
                        snippet = re.sub(r"<.*?>", "", item.get("snippet", "")).strip()
                        page_id = item.get("pageid", "")
                        wiki_url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else "https://en.wikipedia.org"

                        if snippet:
                            web_results.append({
                                "content": f"[Wikipedia: {title}]\n{snippet}\nSource: {wiki_url}",
                                "chunk_index": 1100 + i,
                                "title": title,
                                "url": wiki_url,
                                "source_type": "web_search",
                            })
            except Exception as wiki_err:
                logger.debug(f"ResearchAgent Wikipedia error: {wiki_err}")

        return web_results


research_agent = ResearchAgent()
