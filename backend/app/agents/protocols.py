"""
Multi-Agent Communication Protocols and Behavioral Interfaces.

Defines structural Protocols (interfaces) for specialized agents, and imports
the underlying data transfer schemas from the central DTO layer (backend.app.dtos.agent_dto).
"""

from typing import Any, Protocol, runtime_checkable

# Re-export data models from the central DTO layer to maintain backward compatibility
from backend.app.dtos.agent_dto import (
    AgentMessage,
    AgentResult,
    AgentRole,
    AgentTask,
    AgentTrace,
    AgentTraceStep,
)


__all__ = [
    "AgentMessage",
    "AgentProtocol",
    "AgentResult",
    "AgentRole",
    "AgentTask",
    "AgentTrace",
    "AgentTraceStep",
    "CriticProtocol",
    "PlannerProtocol",
    "ResearcherProtocol",
    "RouterProtocol",
    "SynthesizerProtocol",
]


# -----------------------------------------------------------------------------
# Agent Behavioral Interfaces (Python Structural Protocols)
# -----------------------------------------------------------------------------

@runtime_checkable
class AgentProtocol(Protocol):
    """Base interface that all specialized agents implement."""

    async def execute_task_async(self, task: AgentTask, trace: AgentTrace | None = None) -> AgentResult:
        """Executes a delegated task asynchronously and returns a structured AgentResult."""
        ...


@runtime_checkable
class RouterProtocol(Protocol):
    """Interface for AI-driven intent classification agents."""

    async def classify_intent_async(
        self,
        query: str,
        has_document: bool = False,
        document_title: str = "",
        trace: AgentTrace | None = None,
        doc_title: str | None = None,
    ) -> AgentResult:
        """Classifies user intent into DOCUMENT_RAG, WEB_SEARCH, DIRECT_CHAT, or COMPLEX_ANALYTIC."""
        ...


@runtime_checkable
class PlannerProtocol(Protocol):
    """Interface for query decomposition and refinement agents."""

    async def decompose_query_async(self, query: str, trace: AgentTrace | None = None) -> AgentResult:
        """Deconstructs complex multi-intent questions into targeted sub-queries."""
        ...

    async def rewrite_query_async(
        self,
        query: str,
        critique: str | None = None,
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Rewrites a search query incorporating feedback from Critic agent."""
        ...


@runtime_checkable
class ResearcherProtocol(Protocol):
    """Interface for hybrid retrieval and search agents."""

    async def retrieve_documents_async(
        self,
        query: str,
        sub_queries: list[str],
        vector_store: Any,
        original_query: str = "",
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Orchestrates dense vector + sparse keyword search with reranking."""
        ...

    async def web_search_async(
        self,
        query: str,
        max_results: int = 4,
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Autonomous web search fallback via DuckDuckGo and Wikipedia tools."""
        ...


@runtime_checkable
class CriticProtocol(Protocol):
    """Interface for evaluation, CRAG grading, and Self-RAG reflection agents."""

    async def grade_documents_async(
        self,
        query: str,
        documents: list[dict[str, Any]],
        original_query: str = "",
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Evaluates whether retrieved documents are relevant to user query."""
        ...

    async def grade_hallucination_async(
        self,
        query: str,
        answer: str,
        context_chunks: list[dict[str, Any]],
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Evaluates answer groundedness and faithfulness against context."""
        ...


@runtime_checkable
class SynthesizerProtocol(Protocol):
    """Interface for context merging and grounded prompt assembly agents."""

    async def synthesize_context_async(
        self,
        query: str,
        doc_chunks: list[dict[str, Any]],
        web_chunks: list[dict[str, Any]],
        chat_history: list[dict[str, str]],
        vector_store: Any = None,
        user_id: str | None = None,
        document_metadata: dict[str, Any] | None = None,
        crag_relevant: bool = True,
        trace: AgentTrace | None = None,
    ) -> AgentResult:
        """Merges documents, web results, and memories into an isolated grounded prompt."""
        ...
