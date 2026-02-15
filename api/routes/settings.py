"""User settings API endpoints (available to all authenticated users)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_tenant, get_current_user
from app.constants import load_form_categories, save_form_categories

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


class FormSchemaResponse(BaseModel):
    tenant_id: str
    categories: Dict[str, str]
    count: int


class FormSchemaUpdateRequest(BaseModel):
    categories: Dict[str, str] = Field(
        ...,
        description="Full replacement of form categories.",
    )


@router.get("/form-schema", response_model=FormSchemaResponse)
async def get_my_form_schema(
    tenant_id: str = Depends(get_current_tenant),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Get form schema for the authenticated user's tenant."""
    categories = load_form_categories(tenant_id)
    return FormSchemaResponse(
        tenant_id=tenant_id,
        categories=categories,
        count=len(categories),
    )


@router.put("/form-schema", response_model=FormSchemaResponse)
async def update_my_form_schema(
    request: FormSchemaUpdateRequest,
    tenant_id: str = Depends(get_current_tenant),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Update form schema for the authenticated user's tenant."""
    success = save_form_categories(request.categories, tenant_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save form schema",
        )

    logger.info(
        "Form schema updated via settings: tenant=%s codes=%d by=%s",
        tenant_id,
        len(request.categories),
        user.get("sub"),
    )

    return FormSchemaResponse(
        tenant_id=tenant_id,
        categories=request.categories,
        count=len(request.categories),
    )
