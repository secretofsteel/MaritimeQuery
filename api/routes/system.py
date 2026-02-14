"""System status API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_current_tenant, get_current_user
from app.database import get_distinct_doc_count, get_node_count
from app.nodes import NodeRepository
from pydantic import BaseModel

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


# --- Endpoints ---

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Get live system status with counts from SQLite and ChromaDB."""
    
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
    )
