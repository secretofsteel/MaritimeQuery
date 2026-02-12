"""Query endpoint — the core of the Maritime RAG API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_app_state, get_current_user
from app.state import AppState
from app.orchestrator import orchestrated_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


# --- Request / Response Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation context. Omit to auto-create a new session.",
    )
    use_conversation_context: bool = Field(
        default=True,
        description="Whether to use conversation history for follow-up queries.",
    )


class SourceInfo(BaseModel):
    source: str = ""
    title: Optional[str] = None
    section: Optional[str] = None
    doc_type: Optional[str] = None
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    confidence_pct: int = 0
    confidence_level: str = ""
    confidence_note: str = ""
    sources: List[SourceInfo] = []
    num_sources: int = 0
    retrieval_strategy: str = ""
    topic_extracted: Optional[str] = None
    context_turn: int = 0
    context_reset_note: Optional[str] = None


# --- Endpoint ---

@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Run a query against the maritime document library.

    Synchronous endpoint — blocks until full answer is generated.
    SSE streaming available at POST /api/v1/query/stream (Sub-step 1.5).

    Flow:
    1. Resolve or create session
    2. Load conversation context (if session exists)
    3. Run orchestrated query (analysis -> retrieval -> synthesis)
    4. Consume answer stream into complete text
    5. Save user + assistant messages to session
    6. Auto-generate session title on first real query
    7. Return complete response
    """
    # 1. Ensure index is loaded
    if not app_state.is_ready_for_queries():
        if not app_state.ensure_index_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search index not available. No documents indexed.",
            )
        app_state.ensure_retrievers()

    # 2. Resolve session
    session_id = request.session_id
    if session_id:
        # Validate session exists and belongs to this tenant
        session_mgr = app_state.ensure_session_manager()
        session = session_mgr.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        # Load conversation context
        app_state.switch_session(session_id)
    else:
        # Auto-create session
        session_id = app_state.create_new_session()

    # 3. Run orchestrated query
    logger.info(
        "API query: user=%s tenant=%s session=%s query='%s'",
        user.get("sub"), app_state.tenant_id, session_id, request.query[:80],
    )

    result = orchestrated_query(
        app_state,
        request.query,
        use_conversation_context=request.use_conversation_context,
        enable_hierarchical=True,
        status_callback=None,  # No UI callback for API
    )

    # 4. Consume answer stream
    answer_stream = result.get("answer_stream")
    if answer_stream:
        full_answer = "".join(answer_stream)
        result["answer"] = full_answer

    answer = result.get("answer", "")

    # 5. Save messages to session
    app_state.add_message_to_current_session("user", request.query)
    app_state.add_message_to_current_session(
        "assistant",
        answer,
        {
            "query": request.query,
            "topic_extracted": result.get("topic_extracted"),
            "confidence_pct": result.get("confidence_pct"),
            "confidence_level": result.get("confidence_level"),
            "confidence_note": result.get("confidence_note"),
            "sources": result.get("sources"),
            "num_sources": result.get("num_sources"),
            "retriever_type": result.get("retriever_type"),
            "context_mode": result.get("context_mode"),
            "context_turn": result.get("context_turn"),
            "context_reset_note": result.get("context_reset_note"),
        },
    )

    # 6. Auto-generate title if this is the first real query
    session_mgr = app_state.ensure_session_manager()
    session = session_mgr.get_session(session_id)
    if session and session.title == "New Chat":
        retriever_type = result.get("retriever_type", "")
        if retriever_type != "none":
            try:
                session_mgr.auto_generate_title(session_id, request.query, answer)
            except Exception as exc:
                logger.warning("Title generation failed: %s", exc)

    # 7. Build response
    sources_raw = result.get("sources", [])
    sources = [
        SourceInfo(
            source=s.get("source", ""),
            title=s.get("title"),
            section=s.get("section"),
            doc_type=s.get("doc_type"),
            relevance_score=s.get("score"),
        )
        for s in sources_raw
    ]

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        confidence_pct=result.get("confidence_pct", 0),
        confidence_level=result.get("confidence_level", ""),
        confidence_note=result.get("confidence_note", ""),
        sources=sources,
        num_sources=result.get("num_sources", 0),
        retrieval_strategy=result.get("retrieval_strategy", ""),
        topic_extracted=result.get("topic_extracted"),
        context_turn=result.get("context_turn", 0),
        context_reset_note=result.get("context_reset_note"),
    )
