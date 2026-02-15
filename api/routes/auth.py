"""Authentication routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from api.auth import (
    authenticate_user,
    clear_auth_cookie,
    create_access_token,
    load_credentials,
    set_auth_cookie,
)
from api.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    role: str
    display_name: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    """Authenticate user and return JWT access token.

    Sets an httpOnly cookie for browser clients.
    Also returns the token in the JSON body for API clients.
    """
    user = authenticate_user(request.username, request.password)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(
        username=user["username"],
        tenant_id=user["tenant_id"],
        role=user["role"],
        name=user.get("name", user["username"]),
    )

    # Set httpOnly cookie for browser clients
    set_auth_cookie(response, token)

    logger.info("User '%s' logged in (tenant=%s)", user["username"], user["tenant_id"])

    return LoginResponse(
        access_token=token,
        tenant_id=user["tenant_id"],
        role=user["role"],
        display_name=user.get("name", user["username"]),
    )


class UserInfo(BaseModel):
    username: str
    display_name: str
    tenant_id: str
    role: str


@router.get("/me", response_model=UserInfo)
async def get_current_user_info(
    user: dict = Depends(get_current_user),
):
    """Return the current authenticated user's info.

    Used by the React app on startup to check if an existing
    cookie is still valid and populate the auth context.
    """
    return UserInfo(
        username=user["sub"],
        display_name=user.get("name", user["sub"]),
        tenant_id=user["tenant_id"],
        role=user["role"],
    )


@router.post("/logout")
async def logout(response: Response):
    """Clear the auth cookie.

    The JWT itself can't be invalidated (it's stateless),
    but removing the cookie means the browser stops sending it.
    """
    clear_auth_cookie(response)
    return {"status": "logged_out"}


@router.get("/tenants")
async def list_tenants(
    user: dict = Depends(get_current_user),
):
    """List available tenants. Superuser only.

    Reads tenant_ids from users.yaml.
    """
    if user.get("role") != "superuser":
        raise HTTPException(status_code=403, detail="Superuser access required")

    try:
        credentials = load_credentials()
    except Exception:
        return {"tenants": []}

    tenants = {}
    for username, data in credentials.items():
        tid = data.get("tenant_id")
        if tid and tid not in tenants:
            tenants[tid] = data.get("name", tid)

    tenant_list = [
        {"tenant_id": tid, "display_name": name}
        for tid, name in sorted(tenants.items())
    ]

    return {"tenants": tenant_list}
