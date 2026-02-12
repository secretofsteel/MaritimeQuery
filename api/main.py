"""FastAPI application entry point with lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import AppConfig
from app.indexing import load_cached_nodes_and_index
from app.database import get_node_count

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle.

    Startup:
        1. Initialize AppConfig singleton (API keys, paths, LlamaIndex settings).
        2. Connect to ChromaDB and build VectorStoreIndex.
        3. Store shared resources on app.state for dependency injection.

    Shutdown:
        Log clean shutdown. No explicit resource cleanup needed.
    """
    # --- Startup ---
    logger.info("Starting Maritime RAG API...")

    # 1. Config singleton — loads API keys, configures embedding model
    config = AppConfig.get()
    logger.info("Config loaded: base_dir=%s", config.paths.base_dir)

    # 2. Connect to ChromaDB index
    nodes, index = load_cached_nodes_and_index()

    # 3. Get ChromaDB collection reference for sharing with retrievers
    chroma_collection = None
    if index is not None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(config.paths.chroma_path))
            chroma_collection = client.get_or_create_collection("maritime_docs")
            logger.info(
                "ChromaDB connected: %d vectors in collection",
                chroma_collection.count(),
            )
        except Exception as exc:
            logger.error("Failed to connect to ChromaDB: %s", exc)

    # 4. Store shared resources on app.state
    app.state.config = config
    app.state.index = index
    app.state.chroma_collection = chroma_collection
    app.state.node_count = get_node_count(None)  # Total across all tenants

    logger.info(
        "Startup complete: index=%s, nodes=%d",
        "loaded" if index is not None else "empty",
        app.state.node_count,
    )

    yield  # --- Application runs ---

    # --- Shutdown ---
    logger.info("Shutting down Maritime RAG API.")


app = FastAPI(
    title="Maritime RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

from api.routes.auth import router as auth_router

app.include_router(auth_router)


# --- Health check (validates lifespan worked) ---

@app.get("/api/v1/health")
async def health():
    """System health check.

    Returns status of shared resources loaded during startup.
    """
    has_index = app.state.index is not None
    has_collection = app.state.chroma_collection is not None
    return {
        "status": "ok" if has_index else "degraded",
        "index_loaded": has_index,
        "chroma_connected": has_collection,
        "node_count": app.state.node_count,
    }
