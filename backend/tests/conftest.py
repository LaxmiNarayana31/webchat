import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Set test environment variables before importing app modules
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["ENVIRONMENT"] = "test"


@pytest.fixture
def mock_gemini_client():
    """Provides a mocked gemini_client for testing agents without live API calls."""
    mock = MagicMock()
    mock.generate_text = MagicMock(return_value='{"route": "DOCUMENT_RAG", "confidence": 0.95, "reasoning": "Document query"}')
    mock.generate_text_async = AsyncMock(return_value='{"grounded": true, "useful": true, "critique": "Well grounded"}')
    return mock


@pytest.fixture
def mock_llm_service():
    """Provides a mocked llm_service for response generation."""
    mock = MagicMock()
    mock.generate_response_async = AsyncMock(return_value={
        "text": "This is a verified answer grounded strictly in the context.",
        "model_used": "gemini-2.0-flash",
        "provider": "gemini",
        "fallback_triggered": False,
        "latency_sec": 0.12,
    })
    return mock


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Sample retrieved context chunks for RAG evaluation."""
    return [
        {
            "chunk_index": 0,
            "content": "Antigravity is an advanced agentic AI coding framework built for Pair Programming.",
            "title": "Antigravity Overview",
            "url": "https://antigravity.ai/docs",
            "score": 0.92,
        },
        {
            "chunk_index": 1,
            "content": "The architecture separates DTOs, Core Logic, and Routes into clean independent layers.",
            "title": "Clean Architecture Guide",
            "url": "https://antigravity.ai/docs/architecture",
            "score": 0.88,
        },
    ]


@pytest.fixture
def mock_vector_store(sample_documents):
    """Provides a mock vector store with query capabilities."""
    vs = MagicMock()
    doc_mock_1 = MagicMock()
    doc_mock_1.page_content = sample_documents[0]["content"]
    doc_mock_1.metadata = {"title": sample_documents[0]["title"], "url": sample_documents[0]["url"]}

    doc_mock_2 = MagicMock()
    doc_mock_2.page_content = sample_documents[1]["content"]
    doc_mock_2.metadata = {"title": sample_documents[1]["title"], "url": sample_documents[1]["url"]}

    vs.similarity_search = MagicMock(return_value=[doc_mock_1, doc_mock_2])
    vs.similarity_search_with_relevance_scores = MagicMock(return_value=[(doc_mock_1, 0.92), (doc_mock_2, 0.88)])
    vs.documents = [doc_mock_1, doc_mock_2]
    vs.metadata = {"title": "Antigravity Overview", "url": "https://antigravity.ai/docs"}
    return vs


@pytest.fixture
def api_client():
    """Provides a FastAPI TestClient configured for endpoint integration testing."""
    from backend.main import app
    with TestClient(app) as client:
        yield client
