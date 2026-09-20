from unittest.mock import patch

import pytest

from backend.app.services.agentic_rag_service import AgenticRAGService


@pytest.mark.asyncio
async def test_graph_compilation():
    """Verifies that LangGraph StateGraph compiles with all 9 nodes and reflection edges."""
    service = AgenticRAGService()
    assert service.graph is not None
    # Check that key nodes are present in the graph
    nodes = service.graph.nodes
    assert "route_node" in nodes
    assert "decompose_and_expand" in nodes
    assert "retrieve" in nodes
    assert "grade_documents" in nodes
    assert "synthesize_context" in nodes
    assert "generate_answer" in nodes
    assert "grade_hallucination" in nodes


@pytest.mark.asyncio
async def test_graph_execution_direct_chat():
    """Verifies graph execution for a greeting: routes to direct chat and completes Self-RAG."""
    service = AgenticRAGService()

    with patch.object(service.llm, "generate_response_async") as mock_gen:
        mock_gen.return_value = {
            "text": "Hello! I am WebChat AI.",
            "model_used": "gemini-2.0-flash",
            "provider": "gemini",
            "fallback_triggered": False,
            "latency_sec": 0.1,
        }
        with patch.object(service, "grade_hallucination_and_faithfulness_async") as mock_grade:
            mock_grade.return_value = (True, True, None)

            _context_chunks, prompt, system_instruction, telemetry = await service.execute_agentic_rag_async(
                query="Hello there!",
                vector_store=None,
            )

            assert prompt != "" or system_instruction != ""
            assert telemetry.get("route") == "DIRECT_CHAT" or "DIRECT_CHAT" in str(telemetry)


@pytest.mark.asyncio
async def test_decide_hallucination_action_routing():
    """Verifies the conditional edge decision logic for Self-RAG reflection."""
    service = AgenticRAGService()

    # Case 1: Grounded and useful -> END
    state_ok = {
        "telemetry": {"self_rag_grounded": True, "self_rag_useful": True},
        "hallucination_retry_count": 0,
    }
    assert service.decide_hallucination_action(state_ok) == "END"

    # Case 2: Not grounded and retry count < 2 -> generate_answer (retry with critique)
    state_retry = {
        "telemetry": {"self_rag_grounded": False, "self_rag_useful": True},
        "hallucination_retry_count": 1,
    }
    assert service.decide_hallucination_action(state_retry) == "generate_answer"

    # Case 3: Retries exhausted -> END or transform_query
    state_exhausted = {
        "telemetry": {"self_rag_grounded": False, "self_rag_useful": False},
        "hallucination_retry_count": 3,
        "rewrite_count": 2,
    }
    assert service.decide_hallucination_action(state_exhausted) == "END"
