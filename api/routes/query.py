"""Query endpoint — the core of the Maritime RAG API."""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

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


def _sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event.

    Args:
        event: Event type (token, metadata, done, error).
        data: Payload — will be JSON-serialized.

    Returns:
        SSE-formatted string: "event: {type}\ndata: {json}\n\n"
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/query/stream")
def run_query_stream(
    request: QueryRequest,
    app_state: AppState = Depends(get_app_state),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Run a query and stream the answer as Server-Sent Events.

    Same pipeline as POST /query, but answer tokens are streamed
    in real-time instead of waiting for complete generation.

    The stream blocks during analysis and retrieval (2-10s),
    then tokens flow as Gemini generates them.

    Event types: token, metadata, done, error.
    See SSE protocol documentation for event format.
    """

    def generate():
        try:
            # 1. Ensure index is loaded
            if not app_state.is_ready_for_queries():
                if not app_state.ensure_index_loaded():
                    yield _sse_event("error", {
                        "detail": "Search index not available. No documents indexed.",
                    })
                    return
                app_state.ensure_retrievers()

            # 2. Resolve session
            session_id = request.session_id
            if session_id:
                session_mgr = app_state.ensure_session_manager()
                session = session_mgr.get_session(session_id)
                if session is None:
                    yield _sse_event("error", {
                        "detail": f"Session not found: {session_id}",
                    })
                    return
                app_state.switch_session(session_id)
            else:
                session_id = app_state.create_new_session()

            # 3. Run orchestrated query (blocks during analysis + retrieval)
            logger.info(
                "API stream: user=%s tenant=%s session=%s query='%s'",
                user.get("sub"), app_state.tenant_id, session_id, request.query[:80],
            )

            result = orchestrated_query(
                app_state,
                request.query,
                use_conversation_context=request.use_conversation_context,
                enable_hierarchical=True,
                status_callback=None,
            )

            # 4. Stream answer tokens
            answer_stream = result.get("answer_stream")
            full_answer_parts = []

            if answer_stream:
                for chunk in answer_stream:
                    full_answer_parts.append(chunk)
                    yield _sse_event("token", {"text": chunk})

            full_answer = "".join(full_answer_parts) if full_answer_parts else result.get("answer", "")

            # Handle non-streaming answers (greetings, chitchat, direct responses)
            if not full_answer_parts and full_answer:
                yield _sse_event("token", {"text": full_answer})

            # 5. Save messages to session
            app_state.add_message_to_current_session("user", request.query)
            app_state.add_message_to_current_session(
                "assistant",
                full_answer,
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

            # 6. Auto-generate title
            session_mgr = app_state.ensure_session_manager()
            session = session_mgr.get_session(session_id)
            if session and session.title == "New Chat":
                retriever_type = result.get("retriever_type", "")
                if retriever_type != "none":
                    try:
                        session_mgr.auto_generate_title(
                            session_id, request.query, full_answer,
                        )
                    except Exception as exc:
                        logger.warning("Title generation failed: %s", exc)

            # 7. Send metadata
            sources_raw = result.get("sources", [])
            sources = [
                {
                    "source": s.get("source", ""),
                    "title": s.get("title"),
                    "section": s.get("section"),
                    "doc_type": s.get("doc_type"),
                    "relevance_score": s.get("score"),
                }
                for s in sources_raw
            ]

            yield _sse_event("metadata", {
                "session_id": session_id,
                "confidence_pct": result.get("confidence_pct", 0),
                "confidence_level": result.get("confidence_level", ""),
                "confidence_note": result.get("confidence_note", ""),
                "sources": sources,
                "num_sources": result.get("num_sources", 0),
                "retrieval_strategy": result.get("retrieval_strategy", ""),
                "topic_extracted": result.get("topic_extracted"),
                "context_turn": result.get("context_turn", 0),
                "context_reset_note": result.get("context_reset_note"),
            })

            # 8. Terminal event
            yield _sse_event("done", {})

        except Exception as exc:
            logger.error("Streaming query failed: %s", exc, exc_info=True)
            yield _sse_event("error", {"detail": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if proxied
        },
    )
