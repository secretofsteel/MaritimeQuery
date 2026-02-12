"""FastAPI dependency injection for tenant-scoped AppState."""

from __future__ import annotations

import logging

from fastapi import Depends, Query, Request

from app.state import AppState

logger = logging.getLogger(__name__)


async def get_current_tenant(
    tenant_id: str = Query(default="shared", alias="tenant_id"),
) -> str:
    """Extract tenant_id for the current request.

    STUB: Uses query parameter for testing via Swagger UI.
    Replaced in Sub-step 1.3 with JWT token extraction.

    Usage in Swagger: /api/v1/query?tenant_id=union
    """
    return tenant_id


async def get_app_state(
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
) -> AppState:
    """Build a tenant-scoped AppState with shared resources injected.

    This is the primary dependency for all route handlers. Routes use it as:
        async def my_route(app_state: AppState = Depends(get_app_state)):

    FastAPI automatically chains: get_current_tenant -> get_app_state -> route.

    Creates a fresh AppState per request with:
    - tenant_id from authentication
    - Shared index from app startup (avoids re-loading ChromaDB)
    - Shared ChromaDB collection (avoids creating new PersistentClient per request)
    - Tenant-scoped retrievers (FTS5 + Vector, created cheaply per request)

    Conversation context (query_history, topic, turn_count) is NOT loaded here.
    Routes that need it call app_state.switch_session(session_id) explicitly.
    This keeps the dependency lightweight for endpoints that don't need history
    (session listing, document upload, health checks, etc.).
    """
    state = AppState(tenant_id=tenant_id)

    # Inject shared resources from lifespan
    state.index = request.app.state.index
    state._shared_chroma_collection = request.app.state.chroma_collection

    # Build tenant-scoped retrievers (cheap: just stores references + tenant_id)
    if state.index is not None:
        state.ensure_retrievers()

    return state
