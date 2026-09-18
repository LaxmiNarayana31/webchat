"""
Planner Agent — Query Decomposition and Iterative Refinement.

Decomposes complex multi-intent queries into targeted sub-queries for multi-hop
retrieval, and rewrites queries when the Critic agent reports insufficient relevance.
"""

import json
import time
from typing import Any

from backend.app.agents.protocols import AgentResult, AgentRole, AgentTrace
from backend.app.core.logging import logger

_DECOMPOSITION_PROMPT = """You are a query decomposition agent for a document Q&A system.

Your job is to break down a complex user question into 2-3 specific, self-contained sub-questions
that can each be independently searched in a vector database for relevant passages.

Guidelines:
- Each sub-question should target a distinct aspect of the original query.
- Sub-questions should be self-contained (understandable without the original query).
- Preserve the user's original intent and terminology.
- For simple, single-intent questions, return just the original query.

User Query: {query}

Output strictly a JSON object:
{{"sub_queries": ["sub-question 1", "sub-question 2"], "is_complex": true/false, "reasoning": "brief explanation"}}"""

_REWRITE_PROMPT = """You are a query refinement agent. The following user question failed to retrieve
relevant documents from a knowledge base on the first attempt.

Original Question: {query}
Previous Attempt Result: Low relevance documents retrieved.
{critique_section}

Task: Rewrite this query to improve semantic vector search retrieval.
Focus on:
- Core entities and technical keywords
- Clear, unambiguous intent
- Removing filler words and conversational phrasing
- Adding likely synonyms or related terms

Output ONLY the improved query string, nothing else."""


class PlannerAgent:
    """Query decomposition and refinement agent for multi-hop retrieval."""

    def __init__(self, gemini_client=None):
        """Initializes PlannerAgent with optional injected LLM client."""
        if gemini_client is None:
            from backend.app.clients.gemini_client import (
                gemini_client as default_client,
            )
            self.gemini_client = default_client
        else:
            self.gemini_client = gemini_client

    async def decompose_query_async(
        self,
        query: str,
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Decomposes a complex query into sub-queries for multi-hop retrieval.

        Returns AgentResult with output keys: sub_queries, is_complex, reasoning.
        """
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_decompose"

        # Quick check: short or simple queries don't need decomposition
        words = query.strip().split()
        complex_indicators = [
            "compare", "difference between", "vs", "versus", "and also",
            "both", "as well as", "contrast", "pros and cons", "advantages",
        ]
        is_potentially_complex = any(k in query.lower() for k in complex_indicators) and len(words) > 5

        if not is_potentially_complex:
            duration_ms = (time.time() - start) * 1000
            if trace:
                trace.add_step(
                    agent_role=AgentRole.PLANNER,
                    action="skip_decomposition",
                    input_summary=query[:60],
                    output_summary="Single sub-query (simple question)",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.PLANNER,
                output={"sub_queries": [query], "decomposed": False, "is_complex": False, "reasoning": "Simple single-intent query"},
                execution_time_ms=duration_ms,
            )

        # LLM-powered decomposition for complex queries
        try:
            prompt = _DECOMPOSITION_PROMPT.format(query=query)
            resp = await self.gemini_client.generate_text_async(
                model="gemini-3.1-flash-lite",
                prompt=prompt,
                temperature=0.0,
            )
            parsed = self._parse_decomposition(resp)
            if parsed and len(parsed["sub_queries"]) > 1:
                duration_ms = (time.time() - start) * 1000
                logger.info(
                    f"PlannerAgent: Decomposed into {len(parsed['sub_queries'])} sub-queries: {parsed['sub_queries']}"
                )
                if trace:
                    trace.add_step(
                        agent_role=AgentRole.PLANNER,
                        action="decompose_query",
                        input_summary=query[:60],
                        output_summary=f"{len(parsed['sub_queries'])} sub-queries generated",
                        duration_ms=duration_ms,
                    )
                return AgentResult(
                    task_id=task_id,
                    agent_role=AgentRole.PLANNER,
                    output=parsed,
                    execution_time_ms=duration_ms,
                )
        except Exception as e:
            logger.debug(f"PlannerAgent decomposition fallback: {e}")

        # Fallback: return original query
        duration_ms = (time.time() - start) * 1000
        if trace:
            trace.add_step(
                agent_role=AgentRole.PLANNER,
                action="decompose_fallback",
                input_summary=query[:60],
                output_summary="Fallback to original query",
                duration_ms=duration_ms,
            )
        return AgentResult(
            task_id=task_id,
            agent_role=AgentRole.PLANNER,
            output={"sub_queries": [query], "is_complex": False, "reasoning": "Decomposition fallback"},
            execution_time_ms=duration_ms,
        )

    async def rewrite_query_async(
        self,
        query: str,
        critique: str | None = None,
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Rewrites a query to improve retrieval based on Critic feedback.

        Returns AgentResult with output keys: rewritten_query, original_query.
        """
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_rewrite"

        critique_section = f"Critic Feedback: {critique}" if critique else ""

        try:
            prompt = _REWRITE_PROMPT.format(query=query, critique_section=critique_section)
            improved = await self.gemini_client.generate_text_async(
                model="gemini-3.1-flash-lite",
                prompt=prompt,
                temperature=0.2,
            )
            improved = improved.strip()
            # Check if output is JSON formatted
            if improved.startswith("{") and "rewritten_query" in improved:
                try:
                    js = json.loads(improved)
                    rewritten_text = js.get("rewritten_query", improved)
                    expansion_terms = js.get("expansion_terms", [])
                    return AgentResult(
                        task_id=task_id,
                        agent_role=AgentRole.PLANNER,
                        output={
                            "rewritten_query": rewritten_text,
                            "original_query": query,
                            "expansion_terms": expansion_terms,
                        },
                        execution_time_ms=(time.time() - start) * 1000,
                    )
                except Exception:
                    pass

            if improved and improved.lower() != query.lower():
                duration_ms = (time.time() - start) * 1000
                logger.info(f"PlannerAgent: Rewrote query '{query[:40]}...' -> '{improved[:40]}...'")
                if trace:
                    trace.add_step(
                        agent_role=AgentRole.PLANNER,
                        action="rewrite_query",
                        input_summary=query[:60],
                        output_summary=f"Rewritten: {improved[:60]}",
                        duration_ms=duration_ms,
                    )
                return AgentResult(
                    task_id=task_id,
                    agent_role=AgentRole.PLANNER,
                    output={"rewritten_query": improved, "original_query": query, "expansion_terms": []},
                    execution_time_ms=duration_ms,
                )
        except Exception as e:
            logger.debug(f"PlannerAgent query rewrite fallback: {e}")

        duration_ms = (time.time() - start) * 1000
        if trace:
            trace.add_step(
                agent_role=AgentRole.PLANNER,
                action="rewrite_fallback",
                input_summary=query[:60],
                output_summary="Rewrite failed, using original",
                duration_ms=duration_ms,
            )
        return AgentResult(
            task_id=task_id,
            agent_role=AgentRole.PLANNER,
            output={"rewritten_query": query, "original_query": query, "expansion_terms": []},
            execution_time_ms=duration_ms,
        )

    @staticmethod
    def _parse_decomposition(raw_text: str) -> dict[str, Any] | None:
        """Parses structured JSON decomposition from LLM response."""
        try:
            clean = raw_text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()

            data = json.loads(clean)
            if isinstance(data, list):
                return {
                    "sub_queries": [str(q).strip() for q in data if str(q).strip()],
                    "decomposed": len(data) > 1,
                    "is_complex": len(data) > 1,
                    "reasoning": "Decomposed query list",
                }
            sub_queries = data.get("sub_queries", [])
            if isinstance(sub_queries, list) and len(sub_queries) >= 1:
                return {
                    "sub_queries": [str(q).strip() for q in sub_queries if str(q).strip()],
                    "decomposed": len(sub_queries) > 1,
                    "is_complex": bool(data.get("is_complex", len(sub_queries) > 1)),
                    "reasoning": data.get("reasoning", ""),
                }
            return None
        except Exception:
            return None


planner_agent = PlannerAgent()
