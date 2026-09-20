from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    """Provides a synchronous FastAPI TestClient."""
    return TestClient(app)


def test_health_check_endpoint(client):
    """Verifies that the health endpoint returns 200 and healthy status."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_agent_capabilities_endpoint(client):
    """Verifies that GET /api/agent/capabilities returns all 6 specialized agents."""
    resp = client.get("/api/agent/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 6

    roles = [agent["role"] for agent in data]
    assert "supervisor" in roles
    assert "router" in roles
    assert "planner" in roles
    assert "researcher" in roles
    assert "critic" in roles
    assert "synthesizer" in roles


def test_agent_route_endpoint(client):
    """Verifies that POST /api/agent/route correctly classifies intent."""
    with patch("backend.app.api.agent_routes.router_agent.classify_intent_async", new=AsyncMock()) as mock_classify:
        mock_classify.return_value = {
            "route": "DIRECT_CHAT",
            "confidence": 0.98,
            "reasoning": "Greeting detected",
            "latency_ms": 1.2,
        }

        resp = client.post("/api/agent/route", json={"query": "Hello there!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["route"] == "DIRECT_CHAT"
        assert data["confidence"] == 0.98


def test_agent_query_endpoint(client):
    """Verifies that POST /api/agent/query delegates to supervisor and returns full DTO."""
    mock_workflow = MagicMock()
    mock_workflow.answer = "Antigravity enforces clean separation between DTOs, Core Logic, and Routes."
    mock_workflow.route = "DOCUMENT_RAG"
    mock_workflow.citations = [{"chunk_index": 0, "content": "Sample content", "title": "Doc", "url": "https://a.ai"}]
    mock_workflow.model_used = "gemini-2.0-flash"
    mock_workflow.provider = "gemini"
    mock_workflow.fallback_triggered = False
    mock_workflow.latency_sec = 0.35
    mock_workflow.trace = None
    mock_workflow.telemetry = {"crag_relevant": True}

    with patch("backend.app.api.agent_routes.supervisor_agent.run_workflow_async", new=AsyncMock(return_value=mock_workflow)):
        resp = client.post(
            "/api/agent/query",
            json={
                "query": "Explain the architecture",
                "include_trace": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "clean separation" in data["answer"]
        assert data["route"] == "DOCUMENT_RAG"
        assert len(data["citations"]) == 1
