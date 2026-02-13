"""FastAPI dependency injection for tenant-scoped AppState."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth import decode_token
from app.state import AppState

logger = logging.getLogger(__name__)

# Security scheme — tells Swagger UI to show "Authorize" button
_bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """Extract and verify JWT from Authorization header.

    Returns the full token payload: {sub, tenant_id, role, exp, iat}.

    Raises:
        HTTPException 401: If token is missing, invalid, or expired.
    """
    try:
        payload = decode_token(credentials.credentials)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not payload.get("tenant_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing tenant_id",
        )

    return payload


async def get_current_tenant(
    user: Dict[str, Any] = Depends(get_current_user),
) -> str:
    """Extract tenant_id from verified JWT payload.

    This is the primary tenant dependency. Routes and other dependencies
    use this to get the authenticated tenant context.
    """
    return user["tenant_id"]


async def get_target_tenant(
    target_tenant_id: Optional[str] = Query(
        None,
        description="Override tenant (superuser only). Defaults to authenticated user's tenant.",
    ),
    user: Dict[str, Any] = Depends(get_current_user),
) -> str:
    """Resolve effective tenant for admin operations.

    Non-superusers: always their own tenant.
    Superusers: target_tenant_id if provided, else their own tenant.
    """
    own_tenant = user["tenant_id"]

    if target_tenant_id is None:
        return own_tenant

    if user.get("role") != "superuser":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superusers can manage other tenants' documents",
        )

    return target_tenant_id


async def get_app_state(
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
) -> AppState:
    """Build a tenant-scoped AppState with shared resources injected.

    This is the primary dependency for all route handlers. Routes use it as:
        async def my_route(app_state: AppState = Depends(get_app_state)):

    FastAPI automatically chains:
        get_current_user -> get_current_tenant -> get_app_state -> route.

    Creates a fresh AppState per request with:
    - tenant_id from JWT authentication
    - Shared index from app startup (avoids re-loading ChromaDB)
    - Shared ChromaDB collection (avoids creating new PersistentClient per request)
    - Tenant-scoped retrievers (FTS5 + Vector, created cheaply per request)

    Conversation context (query_history, topic, turn_count) is NOT loaded here.
    Routes that need it call app_state.switch_session(session_id) explicitly.
    """
    state = AppState(tenant_id=tenant_id)

    # Inject shared resources from lifespan
    state.index = request.app.state.index
    state._shared_chroma_collection = request.app.state.chroma_collection

    # Build tenant-scoped retrievers (cheap: just stores references + tenant_id)
    if state.index is not None:
        state.ensure_retrievers()

    return state


async def get_admin_app_state(
    request: Request,
    tenant_id: str = Depends(get_target_tenant),
) -> AppState:
    """AppState scoped to the admin's target tenant.

    Same as get_app_state() but uses get_target_tenant() for tenant resolution,
    allowing superusers to operate on other tenants' documents.
    """
    state = AppState(tenant_id=tenant_id)
    state.index = request.app.state.index
    state._shared_chroma_collection = request.app.state.chroma_collection

    if state.index is not None:
        state.ensure_retrievers()

    return state


async def require_superuser(
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency that enforces superuser role.

    Use on admin-only endpoints:
        async def admin_route(user: dict = Depends(require_superuser)):
    """
    if user.get("role") != "superuser":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required",
        )
    return user
