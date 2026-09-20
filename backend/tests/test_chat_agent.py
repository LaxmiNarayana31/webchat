from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.agents.chat_agent import WebChatAgent
from backend.app.dtos.chat_dto import ChatResponseDto


@pytest.mark.asyncio
async def test_chat_agent_answer_query_async():
    """Verifies that WebChatAgent returns a fully populated ChatResponseDto via Supervisor."""
    agent = WebChatAgent()

    mock_result = MagicMock()
    mock_result.answer = "Antigravity enforces clean separation between DTOs, Core Logic, and Routes."
    mock_result.citations = [{"chunk_index": 0, "content": "Clean architecture...", "title": "Arch", "url": "https://a.ai"}]
    mock_result.model_used = "gemini-2.0-flash"
    mock_result.provider = "gemini"
    mock_result.fallback_triggered = False
    mock_result.latency_sec = 0.25
    mock_result.route = "DOCUMENT_RAG"
    mock_result.trace = None

    with patch.object(agent.supervisor, "run_workflow_async", new=AsyncMock(return_value=mock_result)):
        response = await agent.answer_query_async(
            query="Explain the architecture layers",
            document_content="Layer 1: DTO, Layer 2: Core Logic, Layer 3: Routes",
        )

        assert isinstance(response, ChatResponseDto)
        assert "clean separation" in response.answer
        assert response.model_used == "gemini-2.0-flash"
        assert response.agent_route == "DOCUMENT_RAG"
        assert len(response.citations) == 1


@pytest.mark.asyncio
async def test_chat_agent_answer_query_stream_events():
    """Verifies that WebChatAgent yields reasoning step events and LLM token chunks."""
    agent = WebChatAgent()

    async def mock_stream_events(*args, **kwargs):
        yield {"type": "step", "step": "route", "title": "Intent Query Routing", "status": "done"}
        yield {"type": "step", "step": "crag", "title": "CRAG Grader", "status": "done"}
        yield {"type": "ready", "context_chunks": [], "prompt": "Hello", "system_instruction": "Act as AI"}

    async def mock_llm_stream(*args, **kwargs):
        yield {"chunk": "Hello", "done": False}
        yield {"chunk": " world!", "done": True}

    with (
        patch.object(agent.supervisor, "orchestrate_stream_async", side_effect=mock_stream_events),
        patch.object(agent.llm, "generate_stream_async", side_effect=mock_llm_stream),
    ):
            events = []
            async for event in agent.answer_query_stream_events_async(query="Hello"):
                events.append(event)

            step_types = [e.get("type") for e in events]
            assert "step" in step_types
            assert "citations" in step_types
            assert "chunk" in step_types
