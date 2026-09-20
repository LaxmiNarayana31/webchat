from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agents.planner_agent import PlannerAgent
from backend.app.agents.protocols import AgentTrace


@pytest.mark.asyncio
async def test_simple_query_bypass_decomposition():
    """Simple single-intent queries should bypass decomposition and return unmodified query."""
    planner = PlannerAgent()
    trace = AgentTrace(query="What is the capital of France?")

    result = await planner.decompose_query_async(
        query="What is the capital of France?",
        trace=trace,
    )

    assert result.success is True
    assert result.output["sub_queries"] == ["What is the capital of France?"]
    assert result.output["decomposed"] is False


@pytest.mark.asyncio
async def test_complex_query_decomposition():
    """Complex multi-faceted queries should be decomposed into targeted sub-queries."""
    planner = PlannerAgent()
    mock_json = '["What is dense vector retrieval?", "What is sparse BM25 retrieval?", "How are dense and sparse results merged?"]'

    with patch.object(planner, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_json)

        result = await planner.decompose_query_async(
            query="Compare dense vector retrieval versus sparse BM25 retrieval and how they merge",
        )

        assert result.success is True
        assert len(result.output["sub_queries"]) == 3
        assert result.output["decomposed"] is True


@pytest.mark.asyncio
async def test_query_rewriting_with_critique():
    """Iterative query rewriting incorporates Critic feedback into an expanded search string."""
    planner = PlannerAgent()
    mock_rewrite = '{"rewritten_query": "Antigravity clean architecture DTO routes separation of concerns", "expansion_terms": ["DTO", "routes", "clean architecture"]}'

    with patch.object(planner, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_rewrite)

        result = await planner.rewrite_query_async(
            query="Tell me about architecture",
            critique="Retrieved documents lacked specificity regarding DTO and route layer decoupling.",
        )

        assert result.success is True
        assert "clean architecture" in result.output["rewritten_query"].lower()
        assert len(result.output["expansion_terms"]) > 0


@pytest.mark.asyncio
async def test_planner_fallback_on_llm_error():
    """Planner falls back to rule-based decomposition if LLM call fails."""
    planner = PlannerAgent()

    with patch.object(planner, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(side_effect=Exception("LLM timeout"))

        result = await planner.decompose_query_async(
            query="Explain the difference between router agent versus critic agent",
        )

        assert result.success is True
        assert len(result.output["sub_queries"]) >= 1
