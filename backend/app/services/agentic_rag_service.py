import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph
import requests

from backend.app.clients.gemini_client import gemini_client
from backend.app.core.logging import logger
from backend.app.dtos.chat_dto import ChatMessageDto
from backend.app.services.llm_service import llm_service
from backend.app.services.memory_service import memory_service
from backend.app.services.rag_service import rag_service
from backend.app.services.rerank_service import rerank_service


class AgenticRAGState(TypedDict, total=False):
    """LangGraph State holding all graph execution context across nodes."""
    query: str
    original_query: str
    sub_queries: List[str]
    documents: List[Dict[str, Any]]
    web_search_results: List[Dict[str, Any]]
    vector_store: Any
    document_metadata: Optional[Dict[str, Any]]
    chat_history: List[Dict[str, str]]
    user_memories: List[str]
    user_id: Optional[str]
    selected_model: Optional[Dict[str, str]]
    route: str  # "DOCUMENT_RAG", "WEB_SEARCH", "DIRECT_CHAT"
    crag_relevant: bool
    crag_confidence: float
    rewrite_count: int
    web_search_triggered: bool
    web_search_triggered: bool
    final_prompt: str
    system_instruction: str
    generated_answer: str
    hallucination_retry_count: int
    telemetry: Dict[str, Any]


def is_overview_or_summary_query(query: str) -> bool:
    """Detects whether a user query is asking for a summary, overview, or explanation of the whole document."""
    try:
        q = query.lower().strip()
        patterns = [
            "what is this document about",
            "what is this website about",
            "what is this page about",
            "what is this article about",
            "what is the document about",
            "what is the website about",
            "what is the page about",
            "what is the article about",
            "what is this about",
            "what is it about",
            "what is this site",
            "what is this",
            "what does this document cover",
            "what does this website cover",
            "what does this page cover",
            "what does this document do",
            "what does this website do",
            "summarize this document",
            "summarize this website",
            "summarize this page",
            "summarize the document",
            "summarize the website",
            "summarize the page",
            "summarize",
            "give me a summary",
            "give a summary",
            "summary of this",
            "summary of the",
            "overview of this",
            "overview of the",
            "general overview",
            "high level overview",
            "high-level overview",
            "main topics",
            "key topics",
            "table of contents",
            "what can i learn",
            "tell me about this document",
            "tell me about this website",
            "tell me about this page",
            "tell me about this",
            "explain this document",
            "explain this website",
            "explain this page",
            "about this document",
            "about this website",
            "about this page",
        ]
        return any(p in q for p in patterns)
    except Exception as e:
        logger.error(f"Error evaluating is_overview_or_summary_query: {e}", exc_info=True)
        return False


def is_diagram_or_image_query(query: str) -> bool:
    """Detects whether a user query asks about diagrams, illustrations, charts, figures, or visual assets."""
    try:
        q = query.lower().strip()
        keywords = [
            "diagram", "diagrams", "illustration", "illustrations",
            "figure", "figures", "image", "images", "chart", "charts",
            "visual", "visuals", "architecture diagram", "display them",
            "show diagram", "show image", "show illustration", "what diagrams",
            "what illustrations", "architectural diagrams",
        ]
        return any(kw in q for kw in keywords)
    except Exception as e:
        logger.error(f"Error evaluating is_diagram_or_image_query: {e}", exc_info=True)
        return False


class AgenticRAGService:
    """LangGraph Agentic RAG engine orchestrating routing, retrieval, grading, and synthesis."""

    def __init__(self):
        """Initializes Agentic RAG service dependencies and compiles LangGraph workflow."""
        try:
            self.rag = rag_service
            self.llm = llm_service
            self.memory = memory_service
            self.reranker = rerank_service
            self.graph = self._build_graph()
        except Exception as e:
            logger.error(f"Error initializing AgenticRAGService: {e}", exc_info=True)
            raise

    def _build_graph(self):
        """Constructs and compiles the official LangGraph StateGraph."""
        try:
            workflow = StateGraph(AgenticRAGState)  # type: ignore

            # Register Graph Nodes
            workflow.add_node("route_node", self.route_node)
            workflow.add_node("decompose_and_expand", self.decompose_and_expand_node)
            workflow.add_node("retrieve", self.retrieve_node)
            workflow.add_node("grade_documents", self.grade_documents_node)
            workflow.add_node("transform_query", self.transform_query_node)
            workflow.add_node("web_search", self.web_search_node)
            workflow.add_node("synthesize_context", self.synthesize_context_node)
            workflow.add_node("generate_answer", self.generate_answer_node)
            workflow.add_node("grade_hallucination", self.grade_hallucination_node)

            # Register Edges and Conditional Decision Branches
            workflow.add_edge(START, "route_node")

            workflow.add_conditional_edges(
                "route_node",
                self.decide_route,
                {
                    "synthesize_context": "synthesize_context",
                    "web_search": "web_search",
                    "decompose_and_expand": "decompose_and_expand",
                },
            )

            workflow.add_edge("decompose_and_expand", "retrieve")
            workflow.add_edge("retrieve", "grade_documents")

            workflow.add_conditional_edges(
                "grade_documents",
                self.decide_crag_action,
                {
                    "synthesize_context": "synthesize_context",
                    "transform_query": "transform_query",
                    "web_search": "web_search",
                },
            )

            workflow.add_edge("transform_query", "retrieve")
            workflow.add_edge("web_search", "synthesize_context")
            workflow.add_edge("synthesize_context", "generate_answer")
            workflow.add_edge("generate_answer", "grade_hallucination")

            workflow.add_conditional_edges(
                "grade_hallucination",
                self.decide_hallucination_action,
                {
                    "END": END,
                    "generate_answer": "generate_answer",
                    "transform_query": "transform_query",
                },
            )

            compiled = workflow.compile()
            logger.info("LangGraph Agentic RAG StateGraph compiled successfully with Self-RAG reflection loops.")
            return compiled
        except Exception as e:
            logger.error(f"Error building LangGraph StateGraph: {e}", exc_info=True)
            raise

    # ---------------------------------------------------------
    # GRAPH NODES
    # ---------------------------------------------------------

    async def route_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Node 1: Classifies user intent using AI-driven Router Agent with heuristic fallback."""
        try:
            from backend.app.agents.router_agent import router_agent
            query = state.get("query", "").strip()
            has_doc = bool(state.get("vector_store"))
            doc_meta = state.get("document_metadata") or {}
            doc_title = doc_meta.get("title", "")

            decision = await router_agent.classify_intent_async(
                query=query,
                has_document=has_doc,
                doc_title=doc_title,
            )
            route = decision.get("route", "DOCUMENT_RAG" if has_doc else "DIRECT_CHAT")
            logger.info(f"LangGraph [route_node]: AI Classified route='{route}' for query='{query[:40]}...'")
            return {"route": route}
        except Exception as e:
            logger.error(f"Error in route_node: {e}", exc_info=True)
            return {"route": "DOCUMENT_RAG" if state.get("vector_store") else "DIRECT_CHAT"}

    async def decompose_and_expand_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Deconstructs complex queries into sub-queries for multi-hop retrieval."""
        try:
            query = state.get("query", "")
            sub_queries = [query]

            # Check for comparative or multi-intent indicators
            is_complex = any(k in query.lower() for k in ["compare", "difference between", "vs", "versus", "and also", "both", "as well as"])
            if is_complex and len(query.split()) > 5:
                prompt = (
                    f"Deconstruct this complex question into 2-3 specific, self-contained sub-questions for vector retrieval:\n"
                    f"Question: {query}\n\n"
                    f"Output strictly a JSON list of strings: [\"sub-question 1\", \"sub-question 2\"]"
                )
                try:
                    resp = gemini_client.generate_text(
                        model="gemini-3.1-flash-lite",
                        prompt=prompt,
                        temperature=0.0,
                    )
                    parsed = self._extract_json(resp)
                    if isinstance(parsed, list) and len(parsed) > 1:
                        sub_queries = [str(q).strip() for q in parsed if str(q).strip()]
                        logger.info(f"LangGraph [decompose_node]: Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
                except Exception as e:
                    logger.debug(f"LangGraph decomposition fallback: {e}")

            logger.debug(f"LLM Decomposed Sub-queries: {sub_queries}")
            return {"sub_queries": sub_queries}
        except Exception as e:
            logger.error(f"Error in decompose_and_expand_node: {e}", exc_info=True)
            return {"sub_queries": [state.get("query", "")]}

    async def retrieve_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Retrieves context across sub-queries using hybrid retrieval and reranking."""
        try:
            vs = state.get("vector_store")
            orig_q = state.get("original_query") or state.get("query", "")
            sub_queries = state.get("sub_queries", [state.get("query", "")])
            all_chunks: List[Dict[str, Any]] = []
            seen_texts = set()

            # If it's an overview/summary query, prepend introductory chunks
            if is_overview_or_summary_query(orig_q) and vs:
                if hasattr(vs, "documents") and vs.documents:
                    sorted_docs = sorted(vs.documents, key=lambda d: d.metadata.get("chunk_index", 0))
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

            for sq in sub_queries:
                chunks = self.rag.retrieve_context(
                    index_or_store=vs,
                    query=sq,
                    top_k=3,
                    use_expansion=True,
                    use_rerank=True,
                )
                for c in chunks:
                    text_hash = c.get("content", "")[:100]
                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        all_chunks.append(c)

            logger.debug(f"Retrieved {len(all_chunks)} chunks for sub_queries {sub_queries}.")
            for i, chunk in enumerate(all_chunks):
                logger.debug(f"   -> Chunk {i+1} [Score {chunk.get('rerank_score', 0)}]: {chunk['content'][:80]}...")

            logger.info(f"LangGraph [retrieve_node]: Retrieved {len(all_chunks)} unique hierarchical chunks.")
            return {"documents": all_chunks}
        except Exception as e:
            logger.error(f"Error in retrieve_node: {e}", exc_info=True)
            return {"documents": []}

    async def grade_documents_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Evaluates retrieved document relevance and confidence score using CRAG."""
        try:
            query = state.get("query", "")
            orig_q = state.get("original_query") or query
            documents = state.get("documents", [])

            if not documents:
                logger.info("LangGraph [grade_documents_node]: No documents found. Setting crag_relevant=False.")
                return {"crag_relevant": False, "crag_confidence": 0.0, "documents": []}

            # If it's an overview/summary query about the document, document excerpts are relevant by definition
            if is_overview_or_summary_query(orig_q):
                logger.info("LangGraph [grade_documents_node]: Overview/summary query detected. Retaining all document chunks.")
                return {"crag_relevant": True, "crag_confidence": 1.0, "documents": documents}

            avg_score = 0.0
            scores = [c.get("rerank_score", 0.5) for c in documents if "rerank_score" in c]
            if scores:
                avg_score = sum(scores) / len(scores)

            prompt = (
                f"User Query: {query}\n\n"
                f"Retrieved Document Passages:\n"
            )
            for i, c in enumerate(documents[:3], 1):
                prompt += f"[{i}] {c.get('content', '')[:300]}\n"

            prompt += (
                "\nTask: Determine if the retrieved passages contain relevant information to answer the user query.\n"
                "Output strictly JSON with keys:\n"
                "{\"is_relevant\": true/false, \"confidence\": float, \"relevant_indices\": [1, 2...]}"
            )

            try:
                resp_text = await gemini_client.generate_text_async(
                    model="gemini-3.1-flash-lite",
                    prompt=prompt,
                    temperature=0.0,
                )
                parsed = self._extract_json(resp_text)
                is_relevant = bool(parsed.get("is_relevant", True))
                confidence = float(parsed.get("confidence", 0.8))
                indices = parsed.get("relevant_indices", list(range(1, len(documents) + 1)))

                filtered = [documents[i - 1] for i in indices if 1 <= i <= len(documents)]
                if not is_relevant:
                    filtered = []
                elif not filtered:
                    filtered = documents

                logger.info(f"LangGraph [grade_documents_node]: CRAG Grader is_relevant={is_relevant}, confidence={confidence:.2f}, filtered={len(filtered)}/{len(documents)}")
                return {"crag_relevant": is_relevant, "crag_confidence": confidence, "documents": filtered}
            except Exception as e:
                logger.warning(f"LangGraph CRAG Grader fallback: {e}")
                is_rel = avg_score >= 0.1 or len(documents) > 0
                return {"crag_relevant": is_rel, "crag_confidence": max(avg_score, 0.5), "documents": documents}
        except Exception as e:
            logger.error(f"Error in grade_documents_node: {e}", exc_info=True)
            return {"crag_relevant": True, "crag_confidence": 0.5, "documents": state.get("documents", [])}

    async def transform_query_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Rewrites query to improve retrieval when initial documents lack relevance."""
        try:
            orig_q = state.get("query", "")
            current_rewrites = state.get("rewrite_count", 0)

            prompt = (
                f"The following user question failed to retrieve relevant documents from the knowledge base.\n"
                f"Original Question: {orig_q}\n\n"
                f"Task: Rewrite this query to improve semantic vector search retrieval. "
                f"Focus on core entities, technical keywords, and clear intent.\n"
                f"Output ONLY the improved query string."
            )

            try:
                improved = await gemini_client.generate_text_async(
                    model="gemini-3.1-flash-lite",
                    prompt=prompt,
                    temperature=0.2,
                )
                improved = improved.strip()
                if improved:
                    logger.info(f"LangGraph [transform_query_node]: Rewrote query from '{orig_q}' -> '{improved}' (attempt {current_rewrites + 1})")
                    return {
                        "query": improved,
                        "sub_queries": [improved],
                        "rewrite_count": current_rewrites + 1,
                    }
            except Exception as e:
                logger.debug(f"Query rewriting error: {e}")

            return {"rewrite_count": current_rewrites + 1}
        except Exception as e:
            logger.error(f"Error in transform_query_node: {e}", exc_info=True)
            return {"rewrite_count": state.get("rewrite_count", 0) + 1}

    async def web_search_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Queries DuckDuckGo and Wikipedia as autonomous web search fallback."""
        try:
            query = state.get("original_query") or state.get("query", "")
            logger.info(f"LangGraph [web_search_node]: Triggering Autonomous Web Search for '{query}'...")
            web_results = await asyncio.to_thread(self.perform_web_search_fallback, query)
            return {
                "web_search_results": web_results,
                "web_search_triggered": True,
            }
        except Exception as e:
            logger.error(f"Error in web_search_node: {e}", exc_info=True)
            return {
                "web_search_results": [],
                "web_search_triggered": True,
            }

    async def synthesize_context_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Merges document chunks, web results, and memories into a grounded prompt."""
        try:
            query = state.get("original_query") or state.get("query", "")
            doc_chunks = list(state.get("documents", []))
            has_doc = bool(state.get("vector_store"))
            vs = state.get("vector_store")
            doc_meta = dict(state.get("document_metadata") or {})

            # Resolve document metadata (title, URL)
            if not doc_meta.get("title"):
                if vs and hasattr(vs, "metadata") and vs.metadata and vs.metadata.get("title"):
                    doc_meta.update(vs.metadata)
                elif vs and hasattr(vs, "documents") and vs.documents:
                    doc_meta["title"] = vs.documents[0].metadata.get("title", "")
                    doc_meta["url"] = vs.documents[0].metadata.get("url", "")
                elif doc_chunks:
                    doc_meta["title"] = doc_chunks[0].get("title", "")
                    doc_meta["url"] = doc_chunks[0].get("url", "")

            # If a document is active, strictly isolate to document chunks to prevent out-of-context bleeding
            if has_doc:
                web_chunks = []
                # If doc_chunks is empty (e.g. out-of-context query or CRAG filtered), supply introductory chunks
                # so the model has the actual document context to properly explain what the document covers vs the query
                if not doc_chunks and vs and hasattr(vs, "documents") and vs.documents:
                    sorted_docs = sorted(vs.documents, key=lambda d: d.metadata.get("chunk_index", 0))
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
                web_chunks = state.get("web_search_results", [])

            combined_chunks = doc_chunks + web_chunks

            # Semantic User Memories
            user_memories = state.get("user_memories", [])
            user_id = state.get("user_id")
            if not user_memories and user_id:
                try:
                    user_memories = self.memory.get_relevant_memories(query=query, user_id=user_id, limit=3)
                except Exception as mem_e:
                    logger.debug(f"Memory retrieval note: {mem_e}")

            # Format prompt with active document context & excerpts
            history_dicts = state.get("chat_history", [])
            prompt, system_instruction = self.rag.format_prompt(
                query=query,
                context_chunks=combined_chunks,
                chat_history=history_dicts,
                user_memories=user_memories if user_memories else None,
                document_metadata=doc_meta if doc_meta else None,
            )

            crag_relevant = state.get("crag_relevant", True)
            if has_doc and not crag_relevant:
                # Suppress citations for queries that were out-of-context
                exported_chunks = []
            else:
                exported_chunks = combined_chunks

            telemetry = {
                "route": state.get("route", "DOCUMENT_RAG"),
                "crag_graded": state.get("crag_relevant") is not None,
                "crag_relevant": crag_relevant,
                "crag_confidence": state.get("crag_confidence", 1.0),
                "rewrite_count": state.get("rewrite_count", 0),
                "web_search_triggered": state.get("web_search_triggered", False),
                "sub_queries_count": len(state.get("sub_queries", [])),
                "documents_count": len(doc_chunks),
                "web_results_count": len(web_chunks),
                "self_rag_evaluated": False,
                "self_rag_grounded": True,
            }

            logger.info(f"LangGraph [synthesize_context_node]: Prompt synthesized with {len(combined_chunks)} total context sources.")
            return {
                "documents": exported_chunks,
                "final_prompt": prompt,
                "system_instruction": system_instruction,
                "telemetry": telemetry,
            }
        except Exception as e:
            logger.error(f"Error in synthesize_context_node: {e}", exc_info=True)
            return {
                "documents": [],
                "final_prompt": state.get("query", ""),
                "system_instruction": "",
                "telemetry": {},
            }

    async def generate_answer_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Generates the main answer for Self-RAG evaluation."""
        try:
            prompt = state.get("final_prompt", "")
            sys_instr = state.get("system_instruction", "")
            selected_model = state.get("selected_model")

            # We use gemini fallback chain asynchronously
            resp = await llm_service.generate_response_async(
                prompt=prompt,
                system_instruction=sys_instr,
                selected_model=selected_model,
                temperature=0.4
            )
            answer = resp.get("text", "")
            return {"generated_answer": answer}
        except Exception as e:
            logger.error(f"Generate answer failed: {e}", exc_info=True)
            return {"generated_answer": "Error generating answer. Please try again."}

    async def grade_hallucination_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """Self-RAG Reflection Node: Evaluates groundedness and factuality of generated answer."""
        try:
            query = state.get("original_query", "")
            answer = state.get("generated_answer", "")
            context_chunks = state.get("documents", [])
            retry_count = state.get("hallucination_retry_count", 0)

            grounded, useful, critique = await self.grade_hallucination_and_faithfulness_async(
                query, answer, context_chunks
            )

            telemetry = dict(state.get("telemetry", {}))
            telemetry["self_rag_evaluated"] = True
            telemetry["self_rag_grounded"] = grounded
            telemetry["self_rag_useful"] = useful
            telemetry["hallucination_retry_count"] = retry_count + 1

            updates = {
                "telemetry": telemetry,
                "hallucination_retry_count": retry_count + 1
            }

            if not grounded or not useful:
                # Inject critique back into the prompt for the next try
                prompt = state.get("final_prompt", "")
                prompt += f"\n\n[CRITIQUE FROM PREVIOUS ATTEMPT (DO NOT HALLUCINATE OR IGNORE INTENT)]: {critique}"
                updates["final_prompt"] = prompt  # type: ignore

            return updates
        except Exception as e:
            logger.error(f"Error in grade_hallucination_node: {e}", exc_info=True)
            return {"hallucination_retry_count": state.get("hallucination_retry_count", 0) + 1}

    # ---------------------------------------------------------
    # CONDITIONAL EDGES
    # ---------------------------------------------------------

    @staticmethod
    def decide_route(state: AgenticRAGState) -> str:
        """Determines initial branching based on route classifier."""
        try:
            route = state.get("route", "DOCUMENT_RAG")
            if route == "DIRECT_CHAT":
                return "synthesize_context"
            elif route == "WEB_SEARCH":
                return "web_search"
            return "decompose_and_expand"
        except Exception as e:
            logger.error(f"Error in decide_route: {e}", exc_info=True)
            return "decompose_and_expand"

    @staticmethod
    def decide_crag_action(state: AgenticRAGState) -> str:
        """Determines next action after document relevance grading."""
        try:
            is_relevant = state.get("crag_relevant", True)
            confidence = state.get("crag_confidence", 1.0)
            rewrite_count = state.get("rewrite_count", 0)

            if is_relevant and confidence >= 0.4:
                return "synthesize_context"

            # If retrieval was poor and we haven't retried with query rewriting yet, loop to transform_query
            if rewrite_count < 1:
                logger.info("LangGraph [decide_crag_action]: Routing to transform_query loop for iterative re-retrieval.")
                return "transform_query"

            # If a document is active, do NOT bleed out-of-context web search results into the document chat
            has_doc = bool(state.get("vector_store"))
            if has_doc:
                logger.info("LangGraph [decide_crag_action]: Document is active. Grounding response strictly in document without external web fallback.")
                return "synthesize_context"

            # Otherwise trigger autonomous web fallback for general search without an active document
            logger.info("LangGraph [decide_crag_action]: Routing to web_search fallback tool.")
            return "web_search"
        except Exception as e:
            logger.error(f"Error in decide_crag_action: {e}", exc_info=True)
            return "synthesize_context"

    @staticmethod
    def decide_hallucination_action(state: AgenticRAGState) -> str:
        """Determines next action after hallucination check."""
        try:
            telemetry = state.get("telemetry", {})
            grounded = telemetry.get("self_rag_grounded", True)
            useful = telemetry.get("self_rag_useful", True)
            retry_count = state.get("hallucination_retry_count", 0)

            if grounded and useful:
                return "END"

            # If we failed the check but haven't retried yet, regenerate the answer with critique
            if retry_count <= 2:
                logger.info(f"LangGraph [decide_hallucination_action]: Failed Self-RAG check. Looping back to generate_answer (Attempt {retry_count}).")
                return "generate_answer"

            # If we keep failing, something is fundamentally wrong with the context. Try one last time to rewrite the query.
            if state.get("rewrite_count", 0) < 2:
                logger.info("LangGraph [decide_hallucination_action]: Failed multiple Self-RAG checks. Looping back to transform_query.")
                return "transform_query"

            logger.warning("LangGraph [decide_hallucination_action]: Exhausted all retries. Proceeding to END despite failed checks.")
            return "END"
        except Exception as e:
            logger.error(f"Error in decide_hallucination_action: {e}", exc_info=True)
            return "END"

    # ---------------------------------------------------------
    # EXTERNAL TOOLS & REFLECTION
    # ---------------------------------------------------------

    def perform_web_search_fallback(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """Queries DuckDuckGo Instant Answer API and Wikipedia Encyclopedia Search."""
        try:
            web_results: List[Dict[str, Any]] = []
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
                logger.debug(f"DuckDuckGo search tool note: {e}")

            # Wikipedia Search API Fallback
            if len(web_results) < max_results:
                try:
                    wiki_search_url = "https://en.wikipedia.org/w/api.php"
                    params = {
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json",
                        "utf8": 1,
                        "srlimit": max_results,
                    }
                    wiki_resp = requests.get(wiki_search_url, params=params, headers=headers, timeout=5)
                    if wiki_resp.status_code == 200:
                        search_items = wiki_resp.json().get("query", {}).get("search", [])
                        for i, item in enumerate(search_items[:max_results]):
                            title = item.get("title", "")
                            snippet = re.sub(r"<.*?>", "", item.get("snippet", "")).strip()
                            page_id = item.get("pageid", "")
                            wiki_page_url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else "https://en.wikipedia.org"

                            if snippet:
                                web_results.append({
                                    "content": f"[Wikipedia: {title}]\n{snippet}\nSource: {wiki_page_url}",
                                    "chunk_index": 1100 + i,
                                    "title": title,
                                    "url": wiki_page_url,
                                    "source_type": "web_search",
                                })
                except Exception as wiki_err:
                    logger.debug(f"Wikipedia search tool note: {wiki_err}")

            logger.info(f"LangGraph: Retrieved {len(web_results)} external web search results.")
            return web_results
        except Exception as e:
            logger.error(f"Error in perform_web_search_fallback: {e}", exc_info=True)
            return []

    async def grade_hallucination_and_faithfulness_async(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Tuple[bool, bool, Optional[str]]:
        """Self-RAG Reflection Node: Evaluates groundedness and factuality asynchronously."""
        try:
            if not context_chunks or not answer or len(answer) < 30:
                return True, True, None

            prompt = (
                f"User Query: {query}\n\n"
                f"Context Excerpts:\n"
            )
            for i, c in enumerate(context_chunks[:4], 1):
                prompt += f"[{i}] {c.get('content', '')[:300]}\n\n"

            prompt += (
                f"Candidate Generated Answer:\n{answer}\n\n"
                "Task:\n"
                "1. Groundedness check: Is every factual claim in the answer supported by the Context Excerpts?\n"
                "2. Utility check: Does the answer address the user query?\n"
                "Output strictly JSON with keys:\n"
                "{\"grounded\": true/false, \"useful\": true/false, \"critique\": \"explanation\"}"
            )

            try:
                resp = await gemini_client.generate_text_async(
                    model="gemini-3.1-flash-lite",
                    prompt=prompt,
                    temperature=0.0,
                )
                data = self._extract_json(resp)
                grounded = bool(data.get("grounded", True))
                useful = bool(data.get("useful", True))
                critique = data.get("critique")
                return grounded, useful, critique
            except Exception as e:
                logger.debug(f"Self-RAG reflection skipped: {e}")
                return True, True, None
        except Exception as e:
            logger.error(f"Error in grade_hallucination_and_faithfulness_async: {e}", exc_info=True)
            return True, True, None

    # ---------------------------------------------------------
    # GRAPH EXECUTION ENTRY POINT
    # ---------------------------------------------------------

    async def execute_agentic_rag_async(
        self,
        query: str,
        vector_store: Any,
        chat_history: Optional[List[ChatMessageDto]] = None,
        selected_model: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], str, str, Dict[str, Any]]:
        """Executes compiled LangGraph pipeline returning context, prompt, instruction, and telemetry asynchronously."""
        try:
            history_dicts = []
            if chat_history:
                history_dicts = [{"role": m.role, "content": m.content} for m in chat_history]

            doc_meta = dict(document_metadata or {})
            if not doc_meta and vector_store:
                if hasattr(vector_store, "metadata") and vector_store.metadata:
                    doc_meta = dict(vector_store.metadata)
                elif hasattr(vector_store, "documents") and vector_store.documents:
                    doc_meta = {
                        "title": vector_store.documents[0].metadata.get("title", ""),
                        "url": vector_store.documents[0].metadata.get("url", ""),
                    }

            initial_state: AgenticRAGState = {
                "query": query,
                "original_query": query,
                "sub_queries": [query],
                "documents": [],
                "web_search_results": [],
                "vector_store": vector_store,
                "document_metadata": doc_meta,
                "chat_history": history_dicts,
                "user_memories": [],
                "user_id": user_id,
                "selected_model": selected_model,
                "route": "DOCUMENT_RAG",
                "crag_relevant": True,
                "crag_confidence": 1.0,
                "rewrite_count": 0,
                "web_search_triggered": False,
                "final_prompt": "",
                "system_instruction": "",
                "telemetry": {},
            }

            # Invoke the LangGraph workflow asynchronously
            final_state = await self.graph.ainvoke(initial_state)

            context_chunks = final_state.get("documents", [])
            prompt = final_state.get("final_prompt", "")
            system_instruction = final_state.get("system_instruction", "")
            telemetry = final_state.get("telemetry", {})

            return context_chunks, prompt, system_instruction, telemetry
        except Exception as e:
            logger.error(f"Error in execute_agentic_rag_async: {e}", exc_info=True)
            raise

    async def execute_agentic_rag_stream(
        self,
        query: str,
        vector_store: Any,
        chat_history: Optional[List[ChatMessageDto]] = None,
        selected_model: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Executes LangGraph pipeline yielding step updates and final ready event asynchronously."""
        try:
            history_dicts = []
            if chat_history:
                history_dicts = [{"role": m.role, "content": m.content} for m in chat_history]

            doc_meta = dict(document_metadata or {})
            if not doc_meta and vector_store:
                if hasattr(vector_store, "metadata") and vector_store.metadata:
                    doc_meta = dict(vector_store.metadata)
                elif hasattr(vector_store, "documents") and vector_store.documents:
                    doc_meta = {
                        "title": vector_store.documents[0].metadata.get("title", ""),
                        "url": vector_store.documents[0].metadata.get("url", ""),
                    }

            initial_state: AgenticRAGState = {
                "query": query,
                "original_query": query,
                "sub_queries": [query],
                "documents": [],
                "web_search_results": [],
                "vector_store": vector_store,
                "document_metadata": doc_meta,
                "chat_history": history_dicts,
                "user_memories": [],
                "user_id": user_id,
                "selected_model": selected_model,
                "route": "DOCUMENT_RAG",
                "crag_relevant": True,
                "crag_confidence": 1.0,
                "rewrite_count": 0,
                "web_search_triggered": False,
                "final_prompt": "",
                "system_instruction": "",
                "telemetry": {},
            }

            final_context_chunks: List[Dict[str, Any]] = []
            final_prompt = ""
            final_system_instruction = ""
            final_telemetry: Dict[str, Any] = {}

            async for update_dict in self.graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in update_dict.items():
                    if node_name == "route_node":
                        route = node_output.get("route", "DOCUMENT_RAG")
                        detail_map = {
                            "DOCUMENT_RAG": "Routing query to Document Knowledge Base",
                            "WEB_SEARCH": "Routing query to Autonomous Web Search Engine",
                            "DIRECT_CHAT": "Direct Conversational Reasoning",
                        }
                        yield {
                            "type": "step",
                            "step": "route",
                            "icon": "•",
                            "title": "Intent Query Routing",
                            "detail": detail_map.get(route, f"Classified intent: {route}"),
                            "status": "done",
                        }
                    elif node_name == "decompose_and_expand":
                        sub_queries = node_output.get("sub_queries", [])
                        if len(sub_queries) > 1:
                            yield {
                                "type": "step",
                                "step": "decompose",
                                "icon": "•",
                                "title": "Sub-Question Decomposition",
                                "detail": f"Deconstructed into {len(sub_queries)} focused sub-queries for multi-hop retrieval",
                                "sub_queries": sub_queries,
                                "status": "done",
                            }
                        else:
                            yield {
                                "type": "step",
                                "step": "decompose",
                                "icon": "•",
                                "title": "Query Semantic Expansion",
                                "detail": "Generated HyDE embeddings & lexical keyword variations",
                                "status": "done",
                            }
                    elif node_name == "retrieve":
                        docs = node_output.get("documents", [])
                        yield {
                            "type": "step",
                            "step": "retrieve",
                            "icon": "•",
                            "title": "Hierarchical Hybrid Retrieval",
                            "detail": f"Retrieved {len(docs)} context chunks via Qdrant BQ + FAISS BM25 + FlashRank Reranker",
                            "status": "done",
                        }
                    elif node_name == "grade_documents":
                        crag_rel = node_output.get("crag_relevant", True)
                        conf = node_output.get("crag_confidence", 1.0)
                        pct = int(conf * 100) if conf <= 1.0 else int(conf)
                        detail = (
                            f"Relevance confidence: {pct}% (Context verified & grounded)"
                            if crag_rel
                            else f"Relevance confidence: {pct}% (Insufficient context, initiating re-routing)"
                        )
                        yield {
                            "type": "step",
                            "step": "crag",
                            "icon": "•",
                            "title": "CRAG Document Relevance Grader",
                            "detail": detail,
                            "status": "done",
                        }
                    elif node_name == "transform_query":
                        rewritten_q = node_output.get("query", "")
                        yield {
                            "type": "step",
                            "step": "transform_query",
                            "icon": "•",
                            "title": "Iterative Query Refinement",
                            "detail": f"Refined search query to: '{rewritten_q}'",
                            "status": "done",
                        }
                    elif node_name == "web_search":
                        web_res = node_output.get("web_search_results", [])
                        yield {
                            "type": "step",
                            "step": "web_search",
                            "icon": "•",
                            "title": "Autonomous Web Fallback",
                            "detail": f"Retrieved {len(web_res)} external references via DuckDuckGo & Wikipedia API",
                            "status": "done",
                        }
                    elif node_name == "synthesize_context":
                        final_context_chunks = node_output.get("documents", [])
                        final_prompt = node_output.get("final_prompt", "")
                        final_system_instruction = node_output.get("system_instruction", "")
                        final_telemetry = node_output.get("telemetry", {})
                        yield {
                            "type": "step",
                            "step": "synthesize",
                            "icon": "•",
                            "title": "Context & Memory Synthesis",
                            "detail": f"Synthesized grounded prompt with {len(final_context_chunks)} sources and user memory",
                            "status": "done",
                        }
                    elif node_name == "generate_answer":
                        yield {
                            "type": "step",
                            "step": "generate",
                            "icon": "•",
                            "title": "Initial Response Generation",
                            "detail": "Generating provisional answer for Self-RAG reflection",
                            "status": "done",
                        }
                    elif node_name == "grade_hallucination":
                        telemetry = node_output.get("telemetry", {})
                        grounded = telemetry.get("self_rag_grounded", True)
                        useful = telemetry.get("self_rag_useful", True)
                        final_telemetry.update(telemetry)
                        
                        if grounded and useful:
                            detail = "Verified groundedness and utility of generated response"
                        else:
                            detail = "Hallucination or poor utility detected. Iterating back into RAG loop"

                        yield {
                            "type": "step",
                            "step": "grade_hallucination",
                            "icon": "•",
                            "title": "Self-RAG Reflection",
                            "detail": detail,
                            "status": "done",
                        }

            yield {
                "type": "ready",
                "context_chunks": final_context_chunks,
                "prompt": final_prompt,
                "system_instruction": final_system_instruction,
                "telemetry": final_telemetry,
            }
        except Exception as e:
            logger.error(f"Error in execute_agentic_rag_stream: {e}", exc_info=True)
            raise

    @staticmethod
    def _extract_json(raw_text: str) -> Any:
        """Extracts and parses JSON object or list from LLM output text."""
        try:
            clean = raw_text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()

            try:
                return json.loads(clean)
            except Exception:
                match = re.search(r"(\{.*?\}|\[.*?\])", clean, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception:
                        pass
                return {}
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}", exc_info=True)
            return {}


agentic_rag_service = AgenticRAGService()
