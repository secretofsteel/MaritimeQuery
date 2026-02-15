"""Admin-only API endpoints for system management."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_target_tenant, require_superuser
from app.constants import load_form_categories, save_form_categories
from app.feedback import FeedbackSystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# --- Form Schema ---

class FormSchemaResponse(BaseModel):
    tenant_id: str
    categories: Dict[str, str]
    count: int


class FormSchemaUpdateRequest(BaseModel):
    categories: Dict[str, str] = Field(
        ...,
        description="Full replacement of form categories. Keys are codes, values are descriptions.",
    )


@router.get("/form-schema", response_model=FormSchemaResponse)
async def get_form_schema(
    tenant_id: str = Depends(get_target_tenant),
    user: Dict[str, Any] = Depends(require_superuser),
):
    """Get form schema categories for a tenant.

    Falls back to shared categories if tenant-specific ones don't exist.
    """
    categories = load_form_categories(tenant_id)
    return FormSchemaResponse(
        tenant_id=tenant_id,
        categories=categories,
        count=len(categories),
    )


@router.put("/form-schema", response_model=FormSchemaResponse)
async def update_form_schema(
    request: FormSchemaUpdateRequest,
    tenant_id: str = Depends(get_target_tenant),
    user: Dict[str, Any] = Depends(require_superuser),
):
    """Replace form schema categories for a tenant.

    This is a full replacement — the entire categories dict is overwritten.
    To add a single code, read the current schema, add to it, and PUT back.
    """
    success = save_form_categories(request.categories, tenant_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save form schema",
        )

    logger.info(
        "Form schema updated: tenant=%s codes=%d by=%s",
        tenant_id,
        len(request.categories),
        user.get("sub"),
    )

    return FormSchemaResponse(
        tenant_id=tenant_id,
        categories=request.categories,
        count=len(request.categories),
    )


# --- Feedback Analytics ---

class FeedbackAnalyticsResponse(BaseModel):
    tenant_id: str
    analytics: Dict[str, Any]


@router.get("/feedback-analytics", response_model=FeedbackAnalyticsResponse)
async def get_feedback_analytics(
    tenant_id: str = Depends(get_target_tenant),
    user: Dict[str, Any] = Depends(require_superuser),
):
    """Get feedback analytics for a tenant.

    Returns satisfaction rates, confidence calibration, query refinement stats,
    and system recommendations.
    """
    feedback_system = FeedbackSystem(tenant_id=tenant_id)
    try:
        analytics = feedback_system.analyze_feedback()
    except Exception as exc:
        logger.error("Feedback analytics failed: %s", exc)
        analytics = {
            "total_feedback": 0,
            "satisfaction_rate": 0,
            "incorrect_rate": 0,
            "error": str(exc),
        }

    return FeedbackAnalyticsResponse(
        tenant_id=tenant_id,
        analytics=analytics,
    )
