from typing import Optional

from pydantic import BaseModel, Field



class IdentifyUserRequestDto(BaseModel):
    """Payload for user email identification / registration."""
    email: str = Field(..., description="User email address")
    client_id: Optional[str] = Field(default=None, description="Guest client device UUID to associate")


class CreateSessionRequestDto(BaseModel):
    """Payload for explicit conversation session creation."""
    title: Optional[str] = Field(default="New Conversation", description="Session title")
    url: Optional[str] = Field(default=None, description="Target document URL")
    email: Optional[str] = Field(default=None, description="User email")
    client_id: Optional[str] = Field(default=None, description="Guest client ID")


class SessionSummaryDto(BaseModel):
    """Summary item for conversation session lists."""
    session_id: str = Field(..., description="Unique session UUID")
    title: str = Field(..., description="Session title")
    url: Optional[str] = Field(default=None, description="Target website URL")
    user_email: Optional[str] = Field(default=None, description="Associated user email")
    message_count: int = Field(default=0, description="Total turns in conversation")
    created_at: float = Field(..., description="Creation timestamp")
    updated_at: float = Field(..., description="Last updated timestamp")


class UserQuotaStatusDto(BaseModel):
    """Current rate limit quota status for a user or guest."""
    is_guest: bool = Field(..., description="Whether this is an unverified guest session")
    email: Optional[str] = Field(default=None, description="Identified email address")
    daily_limit: Optional[int] = Field(default=50, description="Max daily requests for email user")
    guest_limit: Optional[int] = Field(default=5, description="Max lifetime requests for guest")
    requests_used_today: Optional[int] = Field(default=0, description="Queries used today")
    requests_remaining: int = Field(..., description="Remaining queries available")
    remaining: Optional[int] = Field(default=None, description="Alias for requests_remaining")
    can_request: bool = Field(..., description="True if quota is available")
    requires_email: Optional[bool] = Field(default=False, description="True if guest must enter email")

