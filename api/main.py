"""FastAPI application entry point with lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import AppConfig
from app.indexing import load_cached_nodes_and_index
from app.nodes import get_node_count

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle.

    Startup:
        1. Initialize AppConfig singleton (API keys, paths, LlamaIndex settings).
        2. Connect to Qdrant and build VectorStoreIndex.
        3. Store shared resources on app.state for dependency injection.

    Shutdown:
        Log clean shutdown. No explicit resource cleanup needed.
    """
    # --- Startup ---
    logger.info("Starting Maritime RAG API...")

    # 1. Config singleton — loads API keys, configures embedding model
    config = AppConfig.get()
    logger.info("Config loaded: base_dir=%s", config.paths.base_dir)

    # 5. Initialize PostgreSQL connection pool
    pg_pool = None
    try:
        from app.pg_database import init_pg_pool, check_pg_connection
        pg_pool = init_pg_pool(config.database_url)
        pg_ok = check_pg_connection()
        logger.info("PostgreSQL connected: %s", "OK" if pg_ok else "FAILED")
    except Exception as exc:
        logger.error("Failed to initialize PostgreSQL pool: %s", exc)

    app.state.pg_pool = pg_pool

    # 2. Connect to Qdrant index
    nodes, index = load_cached_nodes_and_index()

    # 3. Get shared Qdrant client for dependency injection
    qdrant_client = None
    qdrant_collection_name = None
    if index is not None:
        try:
            from app.vector_store import get_qdrant_client, ensure_collection
            qdrant_client = get_qdrant_client()
            qdrant_collection_name = ensure_collection(qdrant_client)
            collection_info = qdrant_client.get_collection(qdrant_collection_name)
            logger.info(
                "Qdrant connected: %d vectors in collection '%s'",
                collection_info.points_count,
                qdrant_collection_name,
            )
        except Exception as exc:
            logger.error("Failed to connect to Qdrant: %s", exc)

    # 4. Store shared resources on app.state
    app.state.config = config
    app.state.index = index
    app.state.qdrant_client = qdrant_client
    app.state.qdrant_collection_name = qdrant_collection_name
    app.state.node_count = get_node_count(None)

    logger.info(
        "Startup complete: index=%s, nodes=%d",
        "loaded" if index is not None else "empty",
        app.state.node_count,
    )

    yield  # --- Application runs ---

    # --- Shutdown ---
    from app.pg_database import close_pg_pool
    close_pg_pool()
    logger.info("Shutting down Maritime RAG API.")


app = FastAPI(
    title="Maritime RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes.auth import router as auth_router
from api.routes.query import router as query_router
from api.routes.sessions import router as sessions_router
from api.routes.feedback import router as feedback_router
from api.routes.admin import router as admin_router
from api.routes.settings import router as settings_router
from api.routes.chat import router as chat_router # Added based on usage in include_router
from api.routes.documents import router as documents_router # Moved up for consolidation
from api.routes.system import router as system_router # Moved up for consolidation

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(sessions_router)
app.include_router(feedback_router)
app.include_router(admin_router)
app.include_router(settings_router)
app.include_router(system_router)


# --- Health check (validates lifespan worked) ---

@app.get("/api/v1/health")
async def health():
    """System health check.

    Returns status of shared resources loaded during startup.
    """
    has_index = app.state.index is not None
    has_qdrant = app.state.qdrant_client is not None
    return {
        "status": "ok" if has_index else "degraded",
        "index_loaded": has_index,
        "qdrant_connected": has_qdrant,
        "node_count": app.state.node_count,
    }
