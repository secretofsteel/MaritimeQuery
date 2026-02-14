"""Qdrant vector store connection and collection management."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)

from .config import AppConfig
from .logger import LOGGER

# Collection vector configuration
VECTOR_SIZE = 768           # gemini-embedding-001 output dimensionality
VECTOR_DISTANCE = Distance.COSINE


def get_qdrant_client(url: str | None = None) -> QdrantClient:
    """Create a Qdrant client from config or explicit URL."""
    config = AppConfig.get()
    target_url = url or config.qdrant_url
    client = QdrantClient(url=target_url)
    LOGGER.debug("Qdrant client connected to %s", target_url)
    return client


def ensure_collection(client: QdrantClient, collection_name: str | None = None) -> str:
    """Ensure the maritime_rag collection exists with correct config.
    
    Creates collection + payload indexes if missing.
    No-ops if already exists (safe to call on every startup).
    
    Returns the collection name.
    """
    config = AppConfig.get()
    name = collection_name or config.qdrant_collection

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=VECTOR_DISTANCE,
            ),
        )
        LOGGER.info("Created Qdrant collection '%s' (%d dims, %s)", 
                     name, VECTOR_SIZE, VECTOR_DISTANCE)
        
        # Payload indexes for filtered queries
        client.create_payload_index(
            collection_name=name,
            field_name="tenant_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        LOGGER.info("Created payload indexes on tenant_id, source")
    else:
        LOGGER.debug("Qdrant collection '%s' already exists", name)

    return name
