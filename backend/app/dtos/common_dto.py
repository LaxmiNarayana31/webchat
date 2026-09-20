from typing import Any, Dict, Optional

from pydantic import BaseModel, Field



class HealthResponseDto(BaseModel):
    """System health check and provider readiness diagnostic."""
    status: str = Field(default="healthy", description="Application health status")
    version: str = Field(..., description="Application version")
    gemini_configured: bool = Field(..., description="Whether GEMINI_API_KEY is present")
    groq_configured: bool = Field(..., description="Whether GROQ_API_KEY is present")
    active_models_count: int = Field(..., description="Count of models in fallback chain")


class ErrorResponseDto(BaseModel):
    """Standardized error payload returned across all endpoints."""
    success: bool = Field(default=False)
    error_type: str = Field(..., description="Class name of the error")
    message: str = Field(..., description="Human-readable error description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Supplementary debugging context")
    path: Optional[str] = Field(default=None, description="Request URL path where failure occurred")
