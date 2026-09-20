from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from backend.app.core.logging import logger
from backend.app.dtos.user_dto import (
    CreateSessionRequestDto,
    IdentifyUserRequestDto,
    UserQuotaStatusDto,
)
from backend.app.repositories.session_repository import session_repository
from backend.app.services.user_service import user_service


router = APIRouter(tags=["User & Sessions"])


@router.post("/user/identify")
@router.post("/users/identify", include_in_schema=False)
def identify_user(req: IdentifyUserRequestDto):
    """Identifies or registers user via email and returns quota and sessions."""
    try:
        user_info = user_service.identify_or_register_user(req.email, client_id=req.client_id)
        if req.client_id and req.email:
            session_repository.link_guest_sessions_to_user(guest_client_id=req.client_id, email=req.email)
        sessions = session_repository.list_sessions(email=req.email)
        return {
            "success": True,
            "user": user_info,
            "sessions": sessions,
        }
    except ValueError as ve:
        logger.warning(f"User validation error: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error identifying user '{req.email}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/user/quota", response_model=UserQuotaStatusDto)
@router.get("/users/quota", response_model=UserQuotaStatusDto, include_in_schema=False)
@router.get("/quota", response_model=UserQuotaStatusDto, include_in_schema=False)
def get_quota(email: Optional[str] = Query(None), client_id: Optional[str] = Query(None)):
    """Returns current user / guest rate limit quota status."""
    try:
        return user_service.get_quota_status(email=email, client_id=client_id)
    except Exception as e:
        logger.error(f"Error retrieving quota for email='{email}', client='{client_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sessions")
@router.get("/user/sessions", include_in_schema=False)
def list_sessions(email: Optional[str] = Query(None), client_id: Optional[str] = Query(None)):
    """Lists saved conversation sessions for an email user or guest device."""
    try:
        return session_repository.list_sessions(email=email, guest_client_id=client_id)
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/sessions")
@router.post("/user/sessions", include_in_schema=False)
def create_session(req: CreateSessionRequestDto):
    """Explicitly creates a new chat session."""
    try:
        return session_repository.create_session(
            title=req.title or "New Conversation",
            url=req.url,
            email=req.email,
            guest_client_id=req.client_id,
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sessions/{session_id}")
@router.get("/user/sessions/{session_id}", include_in_schema=False)
def get_session(session_id: str):
    """Retrieves full conversation messages and citations for a session."""
    try:
        session_data = session_repository.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        return session_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session '{session_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/sessions/{session_id}")
@router.delete("/user/sessions/{session_id}", include_in_schema=False)
def delete_session(session_id: str):
    """Deletes a chat session and all its messages."""
    try:
        deleted = session_repository.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        return {"success": True, "message": "Session deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session '{session_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

