import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.app.dtos.chat_dto import ChatMessageDto, CitationItemDto


# -----------------------------------------------------------------------------
# Agent Roles & Communication DTOs
# -----------------------------------------------------------------------------

class AgentRole(str, Enum):
    """Enumeration of specialized agent roles within the multi-agent system."""
    SUPERVISOR = "supervisor"
    ROUTER = "router"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


# Alias for backward compatibility
AgentRoleDto = AgentRole


class AgentMessage(BaseModel):
    """Structured inter-agent communication message payload."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: AgentRole
    receiver: AgentRole
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    parent_message_id: str | None = None

    def reply(self, sender: AgentRole, action: str, payload: dict[str, Any]) -> "AgentMessage":
        """Creates a reply message linked to this message via parent_message_id."""
        return AgentMessage(
            sender=sender,
            receiver=self.sender,
            action=action,
            payload=payload,
            parent_message_id=self.id,
        )


class AgentTask(BaseModel):
    """Task descriptor delegated from the Supervisor to a specialized agent."""
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    assigned_to: AgentRole
    description: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=5)
    created_at: float = Field(default_factory=time.time)


class AgentResult(BaseModel):
    """Structured result returned by an agent after completing a task."""
    task_id: str
    agent_role: AgentRole
    success: bool = True
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allows dict-like key access delegating to output dictionary."""
        return self.output[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Allows dict-like get method delegating to output dictionary."""
        return self.output.get(key, default)


# -----------------------------------------------------------------------------
# Observability & Trace DTOs
# -----------------------------------------------------------------------------

class AgentTraceStep(BaseModel):
    """Data transfer object for a single step in an agent execution trace."""
    step_index: int = Field(..., description="Sequential index of the execution step")
    agent_role: AgentRole = Field(..., description="Role of the agent that performed this step")
    action: str = Field(..., description="Action or tool invocation performed")
    input_summary: str = Field(default="", description="High-level summary of inputs provided to the agent")
    output_summary: str = Field(default="", description="High-level summary of agent output/decision")
    duration_ms: float = Field(default=0.0, description="Duration of this step in milliseconds")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp when step occurred")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary step metadata or parameters")


AgentTraceStepDto = AgentTraceStep


class AgentTrace(BaseModel):
    """Data transfer object representing the full execution trace across collaborating agents."""
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16], description="Unique trace session ID")
    query: str = Field(..., description="Original user query that triggered the workflow")
    steps: list[AgentTraceStep] = Field(default_factory=list, description="Ordered sequence of executed agent steps")
    total_duration_ms: float = Field(default=0.0, description="Total execution time in milliseconds")
    agent_handoffs: int = Field(default=0, description="Number of handoffs between distinct agent roles")
    started_at: float = Field(default_factory=time.time, description="Unix timestamp when execution started")

    def add_step(
        self,
        agent_role: AgentRole,
        action: str,
        input_summary: str = "",
        output_summary: str = "",
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTraceStep:
        """Appends a new step to the execution trace and increments handoff counter."""
        step = AgentTraceStep(
            step_index=len(self.steps),
            agent_role=agent_role,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self.steps.append(step)
        if len(self.steps) > 1 and self.steps[-2].agent_role != agent_role:
            self.agent_handoffs += 1
        self.total_duration_ms += duration_ms
        return step

    def finalize(self) -> None:
        """Computes final total duration from wall-clock elapsed time."""
        self.total_duration_ms = (time.time() - self.started_at) * 1000


AgentTraceDto = AgentTrace


# -----------------------------------------------------------------------------
# Decision & Capability DTOs
# -----------------------------------------------------------------------------

class AgentRouteDecisionDto(BaseModel):
    """Data transfer object representing an intent routing decision."""
    route: str = Field(..., description="Classified intent route: DOCUMENT_RAG, WEB_SEARCH, DIRECT_CHAT, or COMPLEX_ANALYTIC")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of classification")
    reasoning: str = Field(default="", description="Agent reasoning explaining the routing decision")
    latency_ms: float = Field(default=0.0, description="Latency of routing decision in milliseconds")


class AgentCapabilityDto(BaseModel):
    """Descriptor of an individual agent's capabilities, purpose, and tools."""
    role: str = Field(..., description="Agent role identifier")
    name: str = Field(..., description="Display name of the agent")
    description: str = Field(..., description="Summary of responsibilities and specialized domain")
    autonomous: bool = Field(default=True, description="Whether agent operates autonomously")
    tools: list[str] = Field(default_factory=list, description="List of tools or capabilities available to agent")


# -----------------------------------------------------------------------------
# API Request & Response DTOs
# -----------------------------------------------------------------------------

class AgentQueryRequestDto(BaseModel):
    """Request payload for querying the multi-agent system directly."""
    query: str = Field(..., description="User query or instruction")
    url: str | None = Field(default=None, description="Active document or website URL context")
    document_content: str | None = Field(default=None, description="Raw document text content context")
    chat_history: list[ChatMessageDto] | None = Field(default_factory=list, description="Prior conversation turns")
    selected_model: dict[str, str] | None = Field(default=None, description="Model selection override")
    user_id: str | None = Field(default=None, description="User or session identifier")
    include_trace: bool = Field(default=True, description="Whether to include full execution trace in response")


class AgentQueryResponseDto(BaseModel):
    """Response payload returned by the multi-agent system."""
    answer: str = Field(..., description="Final synthesized and verified answer")
    route: str = Field(..., description="Classified intent route")
    citations: list[CitationItemDto] = Field(default_factory=list, description="Source citations used to ground the answer")
    model_used: str = Field(default="gemini-2.0-flash", description="Model used for generation")
    model_used: str = Field(default="gemini-3.6-flash", description="Model used for generation")
    provider: str = Field(default="gemini", description="Provider used")
    fallback_triggered: bool = Field(default=False, description="Whether fallback model was triggered")
    latency_sec: float = Field(default=0.0, description="Total end-to-end latency in seconds")
    trace: AgentTrace | None = Field(default=None, description="Detailed multi-agent execution trace if requested")
    telemetry: dict[str, Any] = Field(default_factory=dict, description="Execution metrics and confidence scores")
