from unittest.mock import patch

import pytest

from backend.app.agents.protocols import AgentRole, AgentTrace
from backend.app.agents.research_agent import ResearchAgent


@pytest.mark.asyncio
async def test_research_agent_retrieve_documents(mock_vector_store):
    """Verifies that ResearchAgent retrieves and deduplicates chunks across sub-queries."""
    agent = ResearchAgent()
    trace = AgentTrace(query="Antigravity architecture")

    mock_chunks = [
        {"chunk_index": 0, "content": "Antigravity is an agentic framework", "title": "Overview", "url": "https://a.ai", "score": 0.95},
        {"chunk_index": 1, "content": "Clean architecture separates layers", "title": "Arch", "url": "https://a.ai/arch", "score": 0.91},
    ]

    with patch.object(agent.rag, "retrieve_context", return_value=mock_chunks):
        result = await agent.retrieve_documents_async(
            query="Antigravity architecture",
            sub_queries=["Antigravity architecture", "clean architecture layers"],
            vector_store=mock_vector_store,
            trace=trace,
        )

        assert result.success is True
        assert len(result.output["documents"]) == 2
        assert trace.steps[0].agent_role == AgentRole.RESEARCHER


@pytest.mark.asyncio
async def test_research_agent_web_search():
    """Verifies that ResearchAgent executes web search tool when requested."""
    agent = ResearchAgent()
    trace = AgentTrace(query="latest AI news")

    with patch.object(agent, "_perform_web_search") as mock_ws:
        mock_ws.return_value = [
            {
                "content": "Google DeepMind announced new AI advances.",
                "title": "DeepMind News",
                "url": "https://deepmind.google/news",
                "source_type": "web_search",
            }
        ]

        result = await agent.web_search_async(
            query="latest AI news",
            trace=trace,
        )

        assert result.success is True
        assert len(result.output["web_results"]) == 1
        assert "DeepMind" in result.output["web_results"][0]["content"]
        assert trace.steps[0].agent_role == AgentRole.RESEARCHER
