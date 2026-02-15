"""System status API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_current_tenant, get_current_user, require_superuser, get_target_tenant
from api.log_stream import get_log_buffer, register_client, unregister_client
from app.constants import ALLOWED_DOC_TYPES
from app.nodes import get_distinct_doc_count, get_node_count
from app.nodes import NodeRepository
from pydantic import BaseModel
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["system"])

class SystemStatusResponse(BaseModel):
    status: str                    # "ok" | "degraded" | "empty"
    index_loaded: bool
    qdrant_connected: bool
    total_nodes: int               # Live count from SQLite (all tenants)
    total_documents: int           # Live count: distinct doc_ids in SQLite (all tenants)
    qdrant_vectors: int            # Live count from Qdrant collection
    tenant_id: str                 # Requesting user's tenant
    tenant_nodes: int              # Node count for this tenant only
    tenant_documents: int          # Document count for this tenant only
    pg_connected: bool = False
    pg_tables: int = 0


# --- Endpoints ---

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    request: Request,
    tenant_id: str = Depends(get_target_tenant),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Get live system status with counts from SQLite and Qdrant."""
    
    # Live total counts (all tenants)
    total_nodes = get_node_count(None)
    total_docs = get_distinct_doc_count(tenant_id=None)

    # Tenant-specific counts
    repo = NodeRepository(tenant_id=tenant_id)
    tenant_nodes = repo.get_node_count()
    # Optimized: distinct doc count for tenant using DB helper instead of scanning all ids
    tenant_docs = get_distinct_doc_count(tenant_id=tenant_id)

    # Qdrant live count
    qdrant_vectors = 0
    qdrant_client = request.app.state.qdrant_client
    collection_name = request.app.state.qdrant_collection_name
    if qdrant_client and collection_name:
        try:
            info = qdrant_client.get_collection(collection_name)
            qdrant_vectors = info.points_count
        except Exception:
            pass

    has_index = request.app.state.index is not None

    if not has_index:
        status_str = "degraded"
    elif total_nodes == 0:
        status_str = "empty"
    else:
        status_str = "ok"

    from app.pg_database import check_pg_connection, pg_connection

    pg_connected = check_pg_connection()
    pg_tables = 0
    if pg_connected:
        try:
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) AS cnt FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                    row = cur.fetchone()
                    pg_tables = row["cnt"] if row else 0
        except Exception:
            pass

    return SystemStatusResponse(
        status=status_str,
        index_loaded=has_index,
        qdrant_connected=request.app.state.qdrant_client is not None,
        total_nodes=total_nodes,
        total_documents=total_docs,
        qdrant_vectors=qdrant_vectors,
        tenant_id=tenant_id,
        tenant_nodes=tenant_nodes,
        tenant_documents=tenant_docs,
        pg_connected=pg_connected,
        pg_tables=pg_tables,
    )


@router.get("/config/doc-types")
async def get_doc_types(
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Return the list of allowed document types.

    Used by the frontend for filter bars, upload selectors, and metadata editors.
    Single source of truth — add new types in app/constants.py and they appear
    everywhere automatically.
    """
    return {"doc_types": ALLOWED_DOC_TYPES}


@router.get("/logs/stream")
async def stream_logs(
    user: Dict[str, Any] = Depends(require_superuser),
):
    """Stream server logs in real time via SSE. Superuser only.

    On connect, sends the last ~500 buffered log lines (catchup),
    then streams new log entries as they occur.
    """
    import asyncio

    async def generate():
        queue = register_client()
        try:
            # Phase 1: Send buffered history
            for line in get_log_buffer():
                yield f"data: {line}\n\n"

            # Separator so frontend knows catchup is done
            yield f"event: caught_up\ndata: ---\n\n"

            # Phase 2: Stream live
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {entry}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive to prevent connection timeout
                    yield f": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            unregister_client(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
