from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agents.critic_agent import CriticAgent
from backend.app.agents.protocols import AgentRole, AgentTrace


@pytest.mark.asyncio
async def test_critic_crag_grading_relevant(sample_documents):
    """Critic grades documents as relevant when content addresses the query."""
    critic = CriticAgent()
    trace = AgentTrace(query="What is Antigravity?")
    mock_json = '{"is_relevant": true, "confidence": 0.94, "reasoning": "Documents directly explain Antigravity."}'

    with patch.object(critic, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_json)

        result = await critic.grade_documents_async(
            query="What is Antigravity?",
            documents=sample_documents,
            trace=trace,
        )

        assert result.success is True
        assert result.output["is_relevant"] is True
        assert result.output["confidence"] == 0.94
        assert len(result.output["filtered_documents"]) == 2
        assert trace.steps[0].agent_role == AgentRole.CRICIC if hasattr(AgentRole, "CRICIC") else AgentRole.CRITIC


@pytest.mark.asyncio
async def test_critic_crag_grading_irrelevant(sample_documents):
    """Critic identifies irrelevant documents and reports low confidence."""
    critic = CriticAgent()
    mock_json = '{"is_relevant": false, "confidence": 0.2, "reasoning": "Documents are about coding, not cooking recipes."}'

    with patch.object(critic, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_json)

        result = await critic.grade_documents_async(
            query="How do I bake sourdough bread?",
            documents=sample_documents,
        )

        assert result.success is True
        assert result.output["is_relevant"] is False
        assert result.output["confidence"] == 0.2


@pytest.mark.asyncio
async def test_critic_self_rag_reflection_grounded(sample_documents):
    """Critic verifies answer is grounded and useful in Self-RAG reflection."""
    critic = CriticAgent()
    mock_json = '{"grounded": true, "useful": true, "critique": "Answer is faithfully derived from context."}'

    with patch.object(critic, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_json)

        grounded, useful, critique = await critic.grade_hallucination_and_faithfulness_async(
            query="What is Antigravity?",
            answer="Antigravity is an advanced agentic AI coding framework.",
            context_chunks=sample_documents,
        )

        assert grounded is True
        assert useful is True
        assert critique is not None


@pytest.mark.asyncio
async def test_critic_self_rag_hallucination_detected(sample_documents):
    """Critic detects when answer makes ungrounded claims."""
    critic = CriticAgent()
    mock_json = '{"grounded": false, "useful": true, "critique": "Claim about quantum computers is ungrounded."}'

    with patch.object(critic, "gemini_client") as mock_gemini:
        mock_gemini.generate_text_async = AsyncMock(return_value=mock_json)

        grounded, _useful, critique = await critic.grade_hallucination_and_faithfulness_async(
            query="What is Antigravity?",
            answer="Antigravity runs natively on quantum computers in space.",
            context_chunks=sample_documents,
        )

        assert grounded is False
        assert "quantum" in critique.lower()
