
from pydantic import BaseModel, Field



class ModelConfigDto(BaseModel):
    """Configuration mapping for an individual model in the fallback chain."""
    provider: str = Field(..., description="Provider name: 'gemini' or 'groq'")
    model: str = Field(..., description="Model identifier string")


class RateLimitTelemetryDto(BaseModel):
    """Live telemetry stats for a specific model."""
    requests_last_minute: int = Field(default=0, description="Requests recorded in the last 60s sliding window")
    tokens_available: float = Field(default=0.0, description="Remaining tokens in token bucket")
    in_cooldown: bool = Field(default=False, description="Whether model is temporarily cooling down")
    cooldown_remaining_sec: float = Field(default=0.0, description="Seconds remaining in cooldown")
    consecutive_failures: int = Field(default=0, description="Consecutive 429/rate-limit errors recorded")


class ModelCatalogItemDto(BaseModel):
    """Catalog item presenting model metadata and real-time operational status."""
    provider: str = Field(..., description="Provider name ('gemini' or 'groq')")
    model: str = Field(..., description="Model identifier")
    in_cooldown: bool = Field(default=False, description="True if model is cooling down")
    cooldown_remaining_sec: float = Field(default=0.0, description="Cooldown remaining in seconds")
    requests_last_minute: int = Field(default=0, description="Recent request count")
    tokens_available: float = Field(default=0.0, description="Available token bucket capacity")
