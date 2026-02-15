"""Feedback submission endpoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user, get_current_tenant
from app.feedback import FeedbackSystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Feedback submission from the chat UI.

    The frontend sends the query result metadata along with the
    user's feedback. This allows the feedback system to log the
    full context of what the user is rating.
    """
    feedback_type: str = Field(
        ...,
        description="'positive' or 'negative'",
        pattern="^(positive|negative)$",
    )
    correction: str = Field(
        default="",
        max_length=2000,
        description="User's correction text (for negative feedback)",
    )
    # Query result context — passed from the message metadata
    query: str = Field(default="")
    answer: str = Field(default="", max_length=10000)
    confidence_pct: int = Field(default=0)
    confidence_level: str = Field(default="")
    num_sources: int = Field(default=0)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_strategy: str = Field(default="")


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    request: FeedbackRequest,
    tenant_id: str = Depends(get_current_tenant),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Submit feedback for a query response.

    Stores the feedback along with the original query context
    for later analysis.
    """
    feedback_system = FeedbackSystem(tenant_id=tenant_id)

    # Build the result dict that FeedbackSystem.log_feedback expects
    result_context = {
        "query": request.query,
        "answer": request.answer,
        "confidence_pct": request.confidence_pct,
        "confidence_level": request.confidence_level,
        "num_sources": request.num_sources,
        "sources": request.sources,
        "retrieval_strategy": request.retrieval_strategy,
    }

    try:
        feedback_system.log_feedback(
            result=result_context,
            feedback_type=request.feedback_type,
            correction=request.correction,
        )
    except Exception as exc:
        logger.error("Failed to save feedback: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback",
        )

    logger.info(
        "Feedback received: type=%s tenant=%s query='%s'",
        request.feedback_type,
        tenant_id,
        request.query[:60],
    )

    return {"status": "received", "feedback_type": request.feedback_type}
