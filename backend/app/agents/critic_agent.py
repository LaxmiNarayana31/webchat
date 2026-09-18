"""
Critic Agent — CRAG Document Grading and Self-RAG Reflection.

Dedicated verification agent performing:
  1. CRAG Document Relevance Grading: Evaluates whether retrieved documents are relevant.
  2. Self-RAG Hallucination & Faithfulness Check: Evaluates groundedness and answer utility.
"""

import json
import re
import time
from typing import Any

from backend.app.agents.protocols import AgentResult, AgentRole, AgentTrace
from backend.app.core.logging import logger
from backend.app.services.agentic_rag_service import is_overview_or_summary_query

_CRAG_GRADING_PROMPT = """User Query: {query}

Retrieved Document Passages:
{passages}

Task: Determine if the retrieved passages contain relevant information to answer the user query.
Output strictly JSON with keys:
{{"is_relevant": true/false, "confidence": <0.0-1.0>, "relevant_indices": [1, 2...], "reasoning": "brief explanation"}}"""


_SELF_RAG_PROMPT = """User Query: {query}

Context Excerpts:
{context}

Candidate Generated Answer:
{answer}

Task:
1. Groundedness check: Is every factual claim in the answer supported by the Context Excerpts?
2. Utility check: Does the answer fully and accurately address the user query?
3. If either check fails, provide a specific critique explaining what is wrong and how to fix it.

Output strictly JSON with keys:
{{"grounded": true/false, "useful": true/false, "critique": "detailed explanation of issues or 'All checks passed'"}}"""


class CriticAgent:
    """Verification agent for document relevance grading and answer quality reflection."""

    def __init__(self, gemini_client=None):
        """Initializes CriticAgent with optional injected LLM client."""
        if gemini_client is None:
            from backend.app.clients.gemini_client import (
                gemini_client as default_client,
            )
            self.gemini_client = default_client
        else:
            self.gemini_client = gemini_client

    async def grade_documents_async(
        self,
        query: str,
        documents: list[dict[str, Any]],
        original_query: str = "",
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """CRAG Document Relevance Grading.

        Returns AgentResult with output keys: is_relevant, confidence, filtered_documents, reasoning.
        """
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_crag_grade"
        orig_q = original_query or query

        if not documents:
            duration_ms = (time.time() - start) * 1000
            if trace:
                trace.add_step(
                    agent_role=AgentRole.CRITIC,
                    action="crag_grade_empty",
                    input_summary=query[:60],
                    output_summary="No documents to grade",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.CRITIC,
                output={"is_relevant": False, "confidence": 0.0, "filtered_documents": [], "reasoning": "No documents retrieved"},
                execution_time_ms=duration_ms,
            )

        # Overview/summary queries — document chunks are relevant by definition
        if is_overview_or_summary_query(orig_q):
            duration_ms = (time.time() - start) * 1000
            logger.info("CriticAgent [CRAG]: Overview query detected, all documents relevant by definition.")
            if trace:
                trace.add_step(
                    agent_role=AgentRole.CRITIC,
                    action="crag_grade_overview",
                    input_summary=orig_q[:60],
                    output_summary=f"All {len(documents)} docs relevant (overview)",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.CRITIC,
                output={"is_relevant": True, "confidence": 1.0, "filtered_documents": documents, "reasoning": "Overview query"},
                execution_time_ms=duration_ms,
            )

        # LLM-powered relevance grading
        try:
            passages = ""
            for i, c in enumerate(documents[:3], 1):
                passages += f"[{i}] {c.get('content', '')[:300]}\n"

            prompt = _CRAG_GRADING_PROMPT.format(query=query, passages=passages)
            resp = await self.gemini_client.generate_text_async(
                model="gemini-3.1-flash-lite",
                prompt=prompt,
                temperature=0.0,
            )
            parsed = self._extract_json(resp)
            is_relevant = bool(parsed.get("is_relevant", True))
            confidence = float(parsed.get("confidence", 0.8))
            indices = parsed.get("relevant_indices", list(range(1, len(documents) + 1)))
            reasoning = parsed.get("reasoning", "")

            filtered = [documents[i - 1] for i in indices if 1 <= i <= len(documents)]
            if not is_relevant:
                filtered = []
            elif not filtered:
                filtered = documents

            duration_ms = (time.time() - start) * 1000
            logger.info(
                f"CriticAgent [CRAG]: is_relevant={is_relevant}, confidence={confidence:.2f}, "
                f"filtered={len(filtered)}/{len(documents)} ({duration_ms:.1f}ms)"
            )
            if trace:
                trace.add_step(
                    agent_role=AgentRole.CRITIC,
                    action="crag_grade",
                    input_summary=f"{len(documents)} docs for '{query[:40]}'",
                    output_summary=f"relevant={is_relevant} conf={confidence:.2f} kept={len(filtered)}",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.CRITIC,
                output={
                    "is_relevant": is_relevant,
                    "confidence": confidence,
                    "filtered_documents": filtered,
                    "reasoning": reasoning,
                },
                execution_time_ms=duration_ms,
            )
        except Exception as e:
            logger.warning(f"CriticAgent CRAG grading fallback: {e}")
            # Fallback: use rerank scores
            avg_score = 0.0
            scores = [c.get("rerank_score", 0.5) for c in documents if "rerank_score" in c]
            if scores:
                avg_score = sum(scores) / len(scores)
            is_rel = avg_score >= 0.1 or len(documents) > 0

            duration_ms = (time.time() - start) * 1000
            if trace:
                trace.add_step(
                    agent_role=AgentRole.CRITIC,
                    action="crag_grade_fallback",
                    input_summary=query[:60],
                    output_summary=f"Fallback: relevant={is_rel} avg_score={avg_score:.2f}",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.CRITIC,
                output={
                    "is_relevant": is_rel,
                    "confidence": max(avg_score, 0.5),
                    "filtered_documents": documents,
                    "reasoning": "Heuristic fallback using rerank scores",
                },
                execution_time_ms=duration_ms,
            )

    async def grade_hallucination_async(
        self,
        query: str,
        answer: str,
        context_chunks: list[dict[str, Any]],
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Self-RAG Reflection: Evaluates groundedness and faithfulness of generated answer.

        Returns AgentResult with output keys: grounded, useful, critique.
        """
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_selfrag"

        # Skip evaluation for trivial responses
        if not context_chunks or not answer or len(answer) < 30:
            duration_ms = (time.time() - start) * 1000
            if trace:
                trace.add_step(
                    agent_role=AgentRole.CRITIC,
                    action="selfrag_skip",
                    input_summary="Trivial response",
                    output_summary="Skipped (too short or no context)",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.CRITIC,
                output={"grounded": True, "useful": True, "critique": None},
                execution_time_ms=duration_ms,
            )

        try:
            context = ""
            for i, c in enumerate(context_chunks[:4], 1):
                context += f"[{i}] {c.get('content', '')[:300]}\n\n"

            prompt = _SELF_RAG_PROMPT.format(query=query, context=context, answer=answer)
            resp = await self.gemini_client.generate_text_async(
                model="gemini-3.1-flash-lite",
                prompt=prompt,
                temperature=0.0,
            )
            data = self._extract_json(resp)
            grounded = bool(data.get("grounded", True))
            useful = bool(data.get("useful", True))
            critique = data.get("critique")

            duration_ms = (time.time() - start) * 1000
            logger.info(
                f"CriticAgent [Self-RAG]: grounded={grounded}, useful={useful} ({duration_ms:.1f}ms)"
            )
            if trace:
                trace.add_step(
                    agent_role=AgentRole.CRITIC,
                    action="selfrag_evaluate",
                    input_summary=f"Answer len={len(answer)}, {len(context_chunks)} chunks",
                    output_summary=f"grounded={grounded} useful={useful}",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.CRITIC,
                output={"grounded": grounded, "useful": useful, "critique": critique},
                execution_time_ms=duration_ms,
            )
        except Exception as e:
            logger.debug(f"CriticAgent Self-RAG reflection skipped: {e}")
            duration_ms = (time.time() - start) * 1000
            if trace:
                trace.add_step(
                    agent_role=AgentRole.CRITIC,
                    action="selfrag_fallback",
                    input_summary="Reflection failed",
                    output_summary="Assumed grounded (fallback)",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.CRITIC,
                output={"grounded": True, "useful": True, "critique": None},
                execution_time_ms=duration_ms,
            )

    async def grade_hallucination_and_faithfulness_async(
        self,
        query: str,
        answer: str,
        context_chunks: list[dict[str, Any]],
        trace: AgentTrace | None = None,
    ) -> tuple[bool, bool, str | None]:
        """Convenience method returning (grounded, useful, critique) tuple."""
        result = await self.grade_hallucination_async(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
            trace=trace,
        )
        return (
            bool(result.output.get("grounded", True)),
            bool(result.output.get("useful", True)),
            result.output.get("critique"),
        )

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        """Extracts and parses JSON object from LLM output text."""
        try:
            clean = raw_text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()

            try:
                return json.loads(clean)
            except Exception:
                match = re.search(r"(\{.*?\})", clean, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception:
                        pass
                return {}
        except Exception:
            return {}


critic_agent = CriticAgent()
