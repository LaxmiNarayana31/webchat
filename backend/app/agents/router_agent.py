"""
Router Agent — AI-Driven Intent Classification.

Classifies user queries into DOCUMENT_RAG, WEB_SEARCH, DIRECT_CHAT, or COMPLEX_ANALYTIC
using few-shot structured LLM classification with a sub-50ms heuristic fast-path fallback.
"""

import re
import time
from typing import Any

from backend.app.agents.protocols import AgentResult, AgentRole, AgentTrace
from backend.app.core.logging import logger

# Heuristic fast-path patterns for sub-50ms classification without LLM round-trip
_GREETING_PATTERNS = [
    "hi", "hello", "hey", "howdy", "sup", "yo",
    "how are you", "who are you", "what can you do",
    "thank you", "thanks", "good morning", "good evening",
    "good afternoon", "good night", "bye", "goodbye",
]

_WEB_SEARCH_INDICATORS = [
    "latest", "current", "today", "news", "2024", "2025", "2026",
    "price of", "weather in", "stock", "trending", "recently",
    "what happened", "who won", "score", "live",
]

_ROUTER_CLASSIFICATION_PROMPT = """You are an intent classification agent for an AI-powered document Q&A system.

Classify the user's query into exactly ONE of these categories:
- DOCUMENT_RAG: The query is about the content of the currently loaded document/webpage.
- WEB_SEARCH: The query requires real-time or external web knowledge not found in any loaded document.
- DIRECT_CHAT: The query is a greeting, meta-question about the assistant, or casual conversation.
- COMPLEX_ANALYTIC: The query involves comparing, contrasting, or multi-step analysis across the document.

Context:
- Has active document: {has_document}
- Document title: {doc_title}

User Query: {query}

Respond with ONLY a JSON object:
{{"route": "<DOCUMENT_RAG|WEB_SEARCH|DIRECT_CHAT|COMPLEX_ANALYTIC>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""


class RouterAgent:
    """AI-driven intent classification agent with heuristic fast-path fallback."""

    def __init__(self, gemini_client=None):
        """Initializes RouterAgent with optional injected LLM client."""
        if gemini_client is None:
            from backend.app.clients.gemini_client import (
                gemini_client as default_client,
            )
            self.gemini_client = default_client
        else:
            self.gemini_client = gemini_client

    async def classify_intent_async(
        self,
        query: str,
        has_document: bool = False,
        document_title: str = "",
        trace: AgentTrace | None = None,
        doc_title: str | None = None,
    ) -> AgentResult:
        """Classifies user intent using AI with heuristic fast-path fallback.

        Returns an AgentResult with output keys: route, confidence, reasoning, method.
        """
        document_title = doc_title if doc_title is not None else document_title
        start = time.time()
        task_id = (trace.trace_id if trace else "") + "_route"

        # ---------- Fast-path heuristic classification (sub-50ms) ----------
        fast_result = self._heuristic_classify(query, has_document)
        if fast_result and fast_result["confidence"] >= 0.95:
            duration_ms = (time.time() - start) * 1000
            logger.info(
                f"RouterAgent [heuristic]: route={fast_result['route']} "
                f"confidence={fast_result['confidence']:.2f} ({duration_ms:.1f}ms)"
            )
            if trace:
                trace.add_step(
                    agent_role=AgentRole.ROUTER,
                    action="heuristic_classify",
                    input_summary=query[:60],
                    output_summary=f"route={fast_result['route']}",
                    duration_ms=duration_ms,
                )
            return AgentResult(
                task_id=task_id,
                agent_role=AgentRole.ROUTER,
                output={**fast_result, "method": "heuristic"},
                execution_time_ms=duration_ms,
            )

        # ---------- AI-driven LLM classification ----------
        try:
            prompt = _ROUTER_CLASSIFICATION_PROMPT.format(
                has_document=has_document,
                doc_title=document_title or "None",
                query=query,
            )
            resp = await self.gemini_client.generate_text_async(
                model="gemini-3.1-flash-lite",
                prompt=prompt,
                temperature=0.0,
            )
            parsed = self._parse_classification(resp)
            if parsed:
                duration_ms = (time.time() - start) * 1000
                logger.info(
                    f"RouterAgent [AI]: route={parsed['route']} "
                    f"confidence={parsed['confidence']:.2f} reasoning='{parsed.get('reasoning', '')}' ({duration_ms:.1f}ms)"
                )
                if trace:
                    trace.add_step(
                        agent_role=AgentRole.ROUTER,
                        action="ai_classify",
                        input_summary=query[:60],
                        output_summary=f"route={parsed['route']} conf={parsed['confidence']:.2f}",
                        duration_ms=duration_ms,
                    )
                return AgentResult(
                    task_id=task_id,
                    agent_role=AgentRole.ROUTER,
                    output={**parsed, "method": "ai"},
                    execution_time_ms=duration_ms,
                )
        except Exception as e:
            logger.warning(f"RouterAgent AI classification failed: {e}. Using heuristic fallback.")

        # ---------- Heuristic Fallback on Failure ----------
        duration_ms = (time.time() - start) * 1000
        fallback = self._heuristic_classify(query, has_document, force=True) or {
            "route": "DOCUMENT_RAG" if has_document else "DIRECT_CHAT",
            "confidence": 0.6,
            "reasoning": "Default heuristic fallback",
        }
        if trace:
            trace.add_step(
                agent_role=AgentRole.ROUTER,
                action="heuristic_fallback",
                input_summary=query[:60],
                output_summary=f"route={fallback['route']}",
                duration_ms=duration_ms,
            )
        return AgentResult(
            task_id=task_id,
            agent_role=AgentRole.ROUTER,
            output={**fallback, "method": "fallback"},
            execution_time_ms=duration_ms,
        )

    def _heuristic_classify(
        self, query: str, has_document: bool, force: bool = False
    ) -> dict[str, Any] | None:
        """Rule-based fast-path classification. Returns None if uncertain (unless force=True)."""
        q = query.lower().strip()
        words = q.split()

        # Greeting detection: high confidence, fast-path
        if any(
            re.search(rf"\b{re.escape(g)}\b", q) for g in _GREETING_PATTERNS
        ) and len(words) <= 6:
            return {"route": "DIRECT_CHAT", "confidence": 0.98, "reasoning": "Greeting detected"}

        # If no document loaded
        if not has_document:
            if any(indicator in q for indicator in _WEB_SEARCH_INDICATORS):
                return {"route": "WEB_SEARCH", "confidence": 0.95, "reasoning": "Temporal/web indicator detected without active document"}
            if len(words) > 3:
                return {"route": "WEB_SEARCH", "confidence": 0.7, "reasoning": "No document, substantive query"}
            return {"route": "DIRECT_CHAT", "confidence": 0.8, "reasoning": "No document, short query"}

        # Document is loaded — default heuristic confidence is 0.7 (allowing AI classification to run unless forced)
        if force:
            return {"route": "DOCUMENT_RAG", "confidence": 0.7, "reasoning": "Document active fallback"}

        return None

    @staticmethod
    def _parse_classification(raw_text: str) -> dict[str, Any] | None:
        """Parses structured JSON classification from LLM response."""
        import json
        try:
            clean = raw_text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()

            data = json.loads(clean)
            route = data.get("route", "").upper()
            valid_routes = {"DOCUMENT_RAG", "WEB_SEARCH", "DIRECT_CHAT", "COMPLEX_ANALYTIC"}
            if route not in valid_routes:
                return None
            return {
                "route": route,
                "confidence": float(data.get("confidence", 0.8)),
                "reasoning": data.get("reasoning", ""),
            }
        except Exception:
            # Try to extract route from plain text
            for route in ["DOCUMENT_RAG", "WEB_SEARCH", "DIRECT_CHAT", "COMPLEX_ANALYTIC"]:
                if route in raw_text.upper():
                    return {"route": route, "confidence": 0.7, "reasoning": "Extracted from text"}
            return None


router_agent = RouterAgent()
