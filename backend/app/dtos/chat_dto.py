from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field



class ChatMessageDto(BaseModel):
    """Represents a single message turn in a conversation."""
    role: str = Field(..., description="Role of the sender: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")
    citations: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Citations used in assistant response")
    model_used: Optional[str] = Field(default=None, description="Model used")
    provider: Optional[str] = Field(default=None, description="Provider used")


class CitationItemDto(BaseModel):
    """Source citation reference extracted during RAG."""
    chunk_index: int = Field(..., description="Sequential index of the document chunk")
    content: str = Field(..., description="Verbatim or summarized text excerpt")
    title: Optional[str] = Field(default="", description="Source document or page title")
    url: Optional[str] = Field(default="", description="Source URL")


class ChatRequestDto(BaseModel):
    """Payload for conversational queries."""
    query: str = Field(..., description="User question or inquiry")
    url: Optional[str] = Field(default=None, description="Active loaded URL context")
    document_content: Optional[str] = Field(default=None, description="Direct text document to query")
    chat_history: Optional[List[ChatMessageDto]] = Field(default_factory=list, description="Prior conversation turns")
    selected_model: Optional[Dict[str, str]] = Field(default=None, description="Model override: {'provider': '...', 'model': '...'}")
    stream: bool = Field(default=False, description="Whether to stream response tokens via Server-Sent Events (SSE)")
    user_email: Optional[str] = Field(default=None, description="User email for quota & persistent history")
    client_id: Optional[str] = Field(default=None, description="Client/Guest device ID for guest quota tracking")
    session_id: Optional[str] = Field(default=None, description="Target chat session ID")


class ChatResponseDto(BaseModel):
    """Non-streaming conversational response payload."""
    answer: str = Field(..., description="Generated answer grounded on context")
    model_used: str = Field(..., description="Exact model name used for generation")
    provider: str = Field(..., description="Provider used: 'gemini' or 'groq'")
    fallback_triggered: bool = Field(default=False, description="True if primary model failed or was rate limited")
    latency_sec: float = Field(default=0.0, description="End-to-end response generation latency")
    citations: List[CitationItemDto] = Field(default_factory=list, description="Grounded source citations")
    session_id: Optional[str] = Field(default=None, description="Active database session ID")
    quota: Optional[Dict[str, Any]] = Field(default=None, description="User / Guest quota status")
    agent_route: Optional[str] = Field(default=None, description="Classified agent intent route")
    agent_trace: Optional[Dict[str, Any]] = Field(default=None, description="Multi-agent execution trace")


class StreamChunkDto(BaseModel):
    """Single token chunk streamed to the client."""
    chunk: str = Field(default="", description="Text token")
    done: bool = Field(default=False, description="Flag indicating if stream has completed")
    model_used: Optional[str] = Field(default=None, description="Model that produced this token")
    provider: Optional[str] = Field(default=None, description="Provider that produced this token")
    fallback_triggered: bool = Field(default=False, description="Whether fallback occurred during streaming")
    latency_sec: float = Field(default=0.0, description="Latency up to this point")
    session_id: Optional[str] = Field(default=None, description="Active database session ID")
    quota: Optional[Dict[str, Any]] = Field(default=None, description="User / Guest quota status")
    agent_route: Optional[str] = Field(default=None, description="Classified agent intent route")
    agent_trace: Optional[Dict[str, Any]] = Field(default=None, description="Multi-agent execution trace if completed")
