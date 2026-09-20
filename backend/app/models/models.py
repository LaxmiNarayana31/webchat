import json
import time
import uuid
from typing import Any, Dict, List

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.models.base import Base



class UserEntity(Base):
    """User account tracking email and daily rate limit query quotas."""
    __tablename__ = "webchat_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(Float, default=time.time, nullable=False)
    last_active_at = Column(Float, default=time.time, nullable=False)
    daily_requests_count = Column(Integer, default=0, nullable=False)
    last_request_date = Column(String(10), default="", nullable=False)  # "YYYY-MM-DD"
    total_requests_count = Column(Integer, default=0, nullable=False)

    sessions = relationship(
        "ChatSessionEntity",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(ChatSessionEntity.updated_at)",
    )


class ChatSessionEntity(Base):
    """Conversation session attached to a registered user or guest."""
    __tablename__ = "webchat_sessions"

    session_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("webchat_users.id", ondelete="CASCADE"), nullable=True, index=True)
    guest_client_id = Column(String(128), nullable=True, index=True)
    title = Column(String(255), default="New Conversation", nullable=False)
    url = Column(Text, nullable=True)
    created_at = Column(Float, default=time.time, nullable=False)
    updated_at = Column(Float, default=time.time, nullable=False)

    user = relationship("UserEntity", back_populates="sessions")
    messages = relationship(
        "ChatMessageEntity",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessageEntity.created_at",
    )


class ChatMessageEntity(Base):
    """Individual message turn storing role, text, and source citations."""
    __tablename__ = "webchat_messages"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(64), ForeignKey("webchat_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(Text, nullable=False)
    model_used = Column(String(128), nullable=True)
    provider = Column(String(64), nullable=True)
    fallback_triggered = Column(Boolean, default=False, nullable=False)
    latency_sec = Column(Float, default=0.0, nullable=False)
    citations_json = Column(Text, nullable=True)
    created_at = Column(Float, default=time.time, nullable=False)

    session = relationship("ChatSessionEntity", back_populates="messages")

    @property
    def citations(self) -> List[Dict[str, Any]]:
        """Deserializes stored JSON citations into a list of dictionaries."""
        if not self.citations_json:
            return []
        try:
            return json.loads(str(self.citations_json))
        except Exception:
            return []

    @citations.setter
    def citations(self, val: List[Dict[str, Any]]):
        """Serializes list of citations to JSON string for database storage."""
        self.citations_json = json.dumps(val) if val else None


class GuestUsageEntity(Base):
    """Tracks free guest query count (up to 5 free requests)."""
    """Tracks guest query count and enforces dual-layer (client_id + ip_address) daily limits."""
    __tablename__ = "webchat_guest_usage"

    client_id = Column(String(128), primary_key=True)
    ip_address = Column(String(64), nullable=True, index=True)
    request_count = Column(Integer, default=0, nullable=False)
    created_at = Column(Float, default=time.time, nullable=False)
    last_active_at = Column(Float, default=time.time, nullable=False)


class UrlCacheEntity(Base):
    """Caches ingested URL content, metadata, and vector storage reference by URL hash."""
    __tablename__ = "webchat_url_cache"

    url_hash = Column(String(64), primary_key=True, index=True)
    url = Column(Text, nullable=False)
    title = Column(String(255), nullable=False, default="Website Content")
    vector_session_id = Column(String(64), nullable=False, index=True)
    word_count = Column(Integer, default=0, nullable=False)
    strategy_used = Column(String(64), default="auto", nullable=False)
    paywall_bypassed = Column(Boolean, default=False, nullable=False)
    metadata_json = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    created_at = Column(Float, default=time.time, nullable=False)
    last_accessed_at = Column(Float, default=time.time, nullable=False)
    access_count = Column(Integer, default=1, nullable=False)

    @property
    def site_metadata(self) -> Dict[str, Any]:
        """Deserializes stored metadata JSON into a dictionary."""
        if not self.metadata_json:
            return {}
        try:
            return json.loads(str(self.metadata_json))
        except Exception:
            return {}

    @site_metadata.setter
    def site_metadata(self, val: Dict[str, Any]):
        """Serializes dictionary to metadata JSON string for database storage."""
        self.metadata_json = json.dumps(val) if val else None



