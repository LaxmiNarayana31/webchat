from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agents.protocols import AgentRole, AgentTrace
from backend.app.agents.router_agent import RouterAgent


@pytest.mark.asyncio
async def test_fast_path_greeting():
    """Verifies that greetings route immediately to DIRECT_CHAT via sub-50ms heuristic."""
    agent = RouterAgent()
    trace = AgentTrace(query="Hello there!")

    result = await agent.classify_intent_async(
        query="Hello there!",
        has_document=True,
        trace=trace,
    )

    assert result["route"] == "DIRECT_CHAT"
    assert result["method"] == "heuristic"
    assert result["confidence"] >= 0.95
    assert len(trace.steps) == 1
    assert trace.steps[0].agent_role == AgentRole.ROUTER


@pytest.mark.asyncio
async def test_fast_path_web_search_indicators():
    """Verifies that queries with temporal indicators route to WEB_SEARCH when no document is active."""
    agent = RouterAgent()
    result = await agent.classify_intent_async(
        query="What is the latest stock price today?",
        has_document=False,
    )

    assert result["route"] == "WEB_SEARCH"
    assert result["confidence"] >= 0.85


@pytest.mark.asyncio
async def test_ai_classification_document_rag():
    """Verifies that domain questions route to DOCUMENT_RAG via LLM classification."""
    agent = RouterAgent()
    mock_response = '{"route": "DOCUMENT_RAG", "confidence": 0.96, "reasoning": "Query is about loaded document."}'

    with patch.object(agent, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_response)

        result = await agent.classify_intent_async(
            query="Explain the three-tier separation of concerns in this project",
            has_document=True,
            doc_title="Architecture Document",
        )

        assert result["route"] == "DOCUMENT_RAG"
        assert result["confidence"] == 0.96
        assert "separation of concerns" in result.get("reasoning", "") or "Query is about" in result.get("reasoning", "")


@pytest.mark.asyncio
async def test_ai_classification_complex_analytic():
    """Verifies that comparison questions route to COMPLEX_ANALYTIC."""
    agent = RouterAgent()
    mock_response = '{"route": "COMPLEX_ANALYTIC", "confidence": 0.91, "reasoning": "Query compares multiple components."}'

    with patch.object(agent, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_response)

        result = await agent.classify_intent_async(
            query="Compare and contrast the RouterAgent with the SupervisorAgent",
            has_document=True,
        )

        assert result["route"] == "COMPLEX_ANALYTIC"
        assert result["confidence"] == 0.91


@pytest.mark.asyncio
async def test_router_resilience_on_llm_failure():
    """Verifies that RouterAgent gracefully defaults to DOCUMENT_RAG or DIRECT_CHAT if LLM errors."""
    agent = RouterAgent()

    with patch.object(agent, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(side_effect=RuntimeError("API quota exceeded"))

        result = await agent.classify_intent_async(
            query="What are the key findings in section 4?",
            has_document=True,
        )

        assert result["route"] == "DOCUMENT_RAG"
        assert result["method"] == "fallback"
