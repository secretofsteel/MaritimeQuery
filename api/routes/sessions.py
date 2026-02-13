"""Session management endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_app_state, get_current_user
from app.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


# --- Response Models ---

class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    last_active: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    count: int


class MessageInfo(BaseModel):
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = {}


class MessageListResponse(BaseModel):
    session_id: str
    messages: List[MessageInfo]
    count: int


class CreateSessionRequest(BaseModel):
    title: str = Field(default="New Chat", max_length=200)


class UpdateSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# --- Endpoints ---

@router.get("", response_model=SessionListResponse)
def list_sessions(
    limit: Optional[int] = None,
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """List all sessions for the authenticated tenant.

    Returns sessions sorted by last_active (most recent first).
    """
    session_mgr = app_state.ensure_session_manager()
    sessions = session_mgr.list_sessions(limit=limit)

    return SessionListResponse(
        sessions=[
            SessionInfo(
                session_id=s.session_id,
                title=s.title,
                created_at=s.created_at.isoformat(),
                last_active=s.last_active.isoformat(),
                message_count=s.message_count,
            )
            for s in sessions
        ],
        count=len(sessions),
    )


@router.post("", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
def create_session(
    request: CreateSessionRequest = CreateSessionRequest(),
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Create a new chat session.

    Note: The query endpoint auto-creates sessions when session_id is omitted,
    so explicit session creation is only needed when you want a custom title
    or want to pre-create before sending queries.
    """
    session_id = app_state.create_new_session(title=request.title)
    session_mgr = app_state.ensure_session_manager()
    session = session_mgr.get_session(session_id)

    return SessionInfo(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        last_active=session.last_active.isoformat(),
        message_count=session.message_count,
    )


@router.get("/{session_id}", response_model=SessionInfo)
def get_session(
    session_id: str,
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Get session details."""
    session_mgr = app_state.ensure_session_manager()
    session = session_mgr.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    return SessionInfo(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        last_active=session.last_active.isoformat(),
        message_count=session.message_count,
    )


@router.patch("/{session_id}", response_model=SessionInfo)
def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Rename a session."""
    session_mgr = app_state.ensure_session_manager()
    session = session_mgr.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session_mgr.update_title(session_id, request.title)

    # Return updated session
    session = session_mgr.get_session(session_id)
    return SessionInfo(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        last_active=session.last_active.isoformat(),
        message_count=session.message_count,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a session and all its messages."""
    session_mgr = app_state.ensure_session_manager()

    deleted = session_mgr.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )


@router.get("/{session_id}/messages", response_model=MessageListResponse)
def get_messages(
    session_id: str,
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Get all messages in a session.

    Messages are returned in chronological order.
    Assistant messages include metadata with sources, confidence, etc.
    """
    session_mgr = app_state.ensure_session_manager()
    session = session_mgr.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    messages = session_mgr.load_messages(session_id)

    return MessageListResponse(
        session_id=session_id,
        messages=[
            MessageInfo(
                role=m.role,
                content=m.content,
                timestamp=m.timestamp.isoformat(),
                metadata=m.metadata,
            )
            for m in messages
        ],
        count=len(messages),
    )
