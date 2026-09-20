import json
import time
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy.orm import joinedload

from backend.app.core.logging import logger
from backend.app.models.models import ChatMessageEntity, ChatSessionEntity, UserEntity
from backend.config.database import get_db_session


def _sanitize_utf8(text: Optional[str]) -> Optional[str]:
    if text is None or not isinstance(text, str):
        return text
    # Strip lone surrogate code points (\ud800-\udfff) which fail utf-8 encoding in database drivers
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


class DatabaseSessionRepository:
    """Database-backed session and conversation history store (PostgreSQL/SQLite)."""

    def create_session(
        self,
        title: str = "New Conversation",
        url: Optional[str] = None,
        email: Optional[str] = None,
        guest_client_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a new conversational session in the database."""
        try:
            now = time.time()
            sid = session_id or str(uuid.uuid4())

            with get_db_session() as session:
                user_id = None
                if email and email.strip():
                    clean_email = email.strip().lower()
                    user = session.query(UserEntity).filter(UserEntity.email == clean_email).first()
                    if not user:
                        user = UserEntity(
                            email=clean_email,
                            created_at=now,
                            last_active_at=now,
                        )
                        session.add(user)
                        session.flush()
                    user_id = user.id

                db_session = ChatSessionEntity(
                    session_id=sid,
                    user_id=user_id,
                    guest_client_id=guest_client_id,
                    title=_sanitize_utf8(title[:250]) if title else "New Conversation",
                    url=_sanitize_utf8(url),
                    created_at=now,
                    updated_at=now,
                )
                session.add(db_session)
                session.flush()

                logger.info(f"DatabaseSessionRepository: Created session '{sid}' (User: {email or guest_client_id})")
                return {
                    "session_id": sid,
                    "title": db_session.title,
                    "url": db_session.url,
                    "user_email": email,
                    "created_at": db_session.created_at,
                    "updated_at": db_session.updated_at,
                    "messages": [],
                }
        except Exception as e:
            logger.error(f"DatabaseSessionRepository: Error creating session: {e}", exc_info=True)
            raise

    def get_or_create_session(
        self,
        session_id: str,
        title: str = "New Conversation",
        url: Optional[str] = None,
        email: Optional[str] = None,
        guest_client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves existing session or creates it if not found."""
        try:
            existing = self.get_session(session_id)
            if existing:
                return existing
            return self.create_session(
                title=title,
                url=url,
                email=email,
                guest_client_id=guest_client_id,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"DatabaseSessionRepository: Error in get_or_create_session: {e}", exc_info=True)
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a full session with its messages and citations."""
        try:
            with get_db_session() as session:
                db_session = (
                    session.query(ChatSessionEntity)
                    .options(joinedload(ChatSessionEntity.messages), joinedload(ChatSessionEntity.user))
                    .filter(ChatSessionEntity.session_id == session_id)
                    .first()
                )
                if not db_session:
                    return None

                user_email = db_session.user.email if db_session.user else None
                messages = [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "model_used": m.model_used,
                        "provider": m.provider,
                        "fallback_triggered": m.fallback_triggered,
                        "latency_sec": m.latency_sec,
                        "citations": m.citations,
                        "created_at": m.created_at,
                    }
                    for m in db_session.messages
                ]

                return {
                    "session_id": db_session.session_id,
                    "title": db_session.title,
                    "url": db_session.url,
                    "user_email": user_email,
                    "created_at": db_session.created_at,
                    "updated_at": db_session.updated_at,
                    "messages": messages,
                }
        except Exception as e:
            logger.error(f"DatabaseSessionRepository: Error getting session '{session_id}': {e}", exc_info=True)
            raise

    def list_sessions(
        self,
        email: Optional[str] = None,
        guest_client_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Lists sessions for a verified email user or guest device client."""
        try:
            with get_db_session() as session:
                if email and email.strip():
                    clean_email = email.strip().lower()
                    user = session.query(UserEntity).filter(UserEntity.email == clean_email).first()
                    if not user:
                        return []
                    query = session.query(ChatSessionEntity).options(joinedload(ChatSessionEntity.user)).filter(ChatSessionEntity.user_id == user.id)
                elif guest_client_id and guest_client_id.strip():
                    query = session.query(ChatSessionEntity).filter(ChatSessionEntity.guest_client_id == guest_client_id.strip())
                else:
                    return []

                db_sessions = query.order_by(ChatSessionEntity.updated_at.desc()).limit(limit).all()

                results = []
                for s in db_sessions:
                    msg_count = session.query(ChatMessageEntity).filter(ChatMessageEntity.session_id == s.session_id).count()
                    if msg_count > 0:
                        results.append({
                            "session_id": s.session_id,
                            "title": s.title,
                            "url": s.url,
                            "user_email": s.user.email if s.user else None,
                            "message_count": msg_count,
                            "created_at": s.created_at,
                            "updated_at": s.updated_at,
                        })
                    else:
                        # Purge empty orphan session with 0 messages
                        try:
                            session.delete(s)
                            session.flush()
                        except Exception:
                            pass
                return results
        except Exception as e:
            logger.error(f"DatabaseSessionRepository: Error listing sessions: {e}", exc_info=True)
            raise

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        provider: Optional[str] = None,
        fallback_triggered: bool = False,
        latency_sec: float = 0.0,
        citations: Optional[List[Dict[str, Any]]] = None,
        url: Optional[str] = None,
        email: Optional[str] = None,
        guest_client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Appends a new message turn to the database session."""
        try:
            now = time.time()
            msg_id = str(uuid.uuid4())

            with get_db_session() as session:
                db_session = session.query(ChatSessionEntity).filter(ChatSessionEntity.session_id == session_id).first()
                if not db_session:
                    user_id = None
                    if email and email.strip():
                        clean_email = email.strip().lower()
                        user = session.query(UserEntity).filter(UserEntity.email == clean_email).first()
                        if user:
                            user_id = user.id

                    title = _sanitize_utf8(content[:60]) if role == "user" else "New Conversation"
                    db_session = ChatSessionEntity(
                        session_id=session_id,
                        user_id=user_id,
                        guest_client_id=guest_client_id,
                        title=title or "New Conversation",
                        url=_sanitize_utf8(url),
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(db_session)
                    session.flush()

                clean_content = _sanitize_utf8(content) or ""
                if db_session.title == "New Conversation" and role == "user" and clean_content.strip():
                    db_session.title = clean_content.strip()[:60] + ("..." if len(clean_content.strip()) > 60 else "")  # type: ignore

                if url and not db_session.url:
                    db_session.url = _sanitize_utf8(url)  # type: ignore

                db_session.updated_at = now  # type: ignore

                citations_str = _sanitize_utf8(json.dumps(citations, default=str)) if citations else None
                msg = ChatMessageEntity(
                    id=msg_id,
                    session_id=session_id,
                    role=role,
                    content=clean_content,
                    model_used=_sanitize_utf8(model_used),
                    provider=_sanitize_utf8(provider),
                    fallback_triggered=fallback_triggered,
                    latency_sec=latency_sec,
                    citations_json=citations_str,
                    created_at=now,
                )
                session.add(msg)
                session.flush()

                return {
                    "id": msg_id,
                    "session_id": session_id,
                    "role": role,
                    "content": clean_content,
                    "model_used": model_used,
                    "provider": provider,
                    "fallback_triggered": fallback_triggered,
                    "latency_sec": latency_sec,
                    "citations": citations or [],
                    "created_at": now,
                }
        except Exception as e:
            logger.error(f"DatabaseSessionRepository: Error appending message to '{session_id}': {e}", exc_info=True)
            raise

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and cascading messages from the database."""
        try:
            with get_db_session() as session:
                db_session = session.query(ChatSessionEntity).filter(ChatSessionEntity.session_id == session_id).first()
                if db_session:
                    session.delete(db_session)
                    logger.info(f"DatabaseSessionRepository: Deleted session '{session_id}'")
                    return True
                return False
        except Exception as e:
            logger.error(f"DatabaseSessionRepository: Error deleting session '{session_id}': {e}", exc_info=True)
            raise

    def link_guest_sessions_to_user(self, guest_client_id: str, email: str) -> int:
        """Links all unassigned guest sessions for a client ID to a verified email user account."""
        try:
            if not guest_client_id or not email:
                return 0
            clean_email = email.strip().lower()
            with get_db_session() as session:
                user = session.query(UserEntity).filter(UserEntity.email == clean_email).first()
                if not user:
                    now = time.time()
                    user = UserEntity(email=clean_email, created_at=now, last_active_at=now)
                    session.add(user)
                    session.flush()

                unlinked = (
                    session.query(ChatSessionEntity)
                    .filter(
                        ChatSessionEntity.guest_client_id == guest_client_id,
                        ChatSessionEntity.user_id.is_(None),
                    )
                    .all()
                )
                count = 0
                for s in unlinked:
                    msg_count = session.query(ChatMessageEntity).filter(ChatMessageEntity.session_id == s.session_id).count()
                    if msg_count > 0:
                        s.user_id = user.id
                        count += 1
                    else:
                        session.delete(s)
                session.flush()
                logger.info(f"DatabaseSessionRepository: Linked {count} guest sessions to user '{clean_email}'")
                return count
        except Exception as e:
            logger.error(f"DatabaseSessionRepository: Error linking guest sessions: {e}", exc_info=True)
            return 0


session_repository = DatabaseSessionRepository()
SessionRepository = DatabaseSessionRepository

