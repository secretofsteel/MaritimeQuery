"""PostgreSQL FTS-based retriever and tenant-aware vector retriever.

Phase 3: PostgreSQL FTS for keyword search (replaces SQLite FTS5)
Phase 4: Tenant-aware vector retrieval (filters Qdrant by tenant)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .pg_database import pg_connection
from .logger import LOGGER


from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)


def _expand_doc_type_case(doc_type_filter: List[str]) -> List[str]:
    """Expand doc_type filter values to cover common case variants.

    Production data contains mixed casing (e.g. ``"Form"`` vs ``"FORM"``,
    ``"Procedure"`` vs ``"PROCEDURE"``).  Rather than requiring a full
    reindex, this helper emits both UPPER and Title variants for each
    requested type so that Qdrant filters match regardless of how
    the metadata was stored.

    Duplicates are removed to keep filter lists clean.
    """
    expanded: set = set()
    for dt in doc_type_filter:
        expanded.add(dt)             # original
        expanded.add(dt.upper())     # FORM, PROCEDURE
        expanded.add(dt.title())     # Form, Procedure
    return sorted(expanded)


# =============================================================================
# QDRANT PAYLOAD HELPERS
# =============================================================================

# Keys written by LlamaIndex internals — not useful as user-facing metadata
_INTERNAL_PAYLOAD_KEYS = frozenset({
    "_node_content", "_node_type", "document_id", "doc_id", "ref_doc_id",
})


def _extract_text_from_payload(payload: dict) -> str:
    """Extract text content from a Qdrant point's payload.

    LlamaIndex stores the full node as JSON in ``_node_content``.
    Falls back to common alternative field names.
    """
    node_content = payload.get("_node_content")
    if node_content:
        try:
            parsed = json.loads(node_content)
            text = parsed.get("text", "")
            if text:
                return text
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: direct payload fields
    return payload.get("text", "") or payload.get("document", "")


def _extract_metadata_from_payload(payload: dict) -> dict:
    """Extract user-facing metadata from a Qdrant point's payload.

    Strips LlamaIndex internal keys so downstream consumers
    (reranker, LLM context) see clean metadata.
    """
    return {k: v for k, v in payload.items() if k not in _INTERNAL_PAYLOAD_KEYS}


# =============================================================================
# POSTGRESQL FTS RETRIEVER
# =============================================================================

class PgFTSRetriever(BaseRetriever):
    """Full-text search retriever using PostgreSQL tsvector/tsquery.

    Replaces SQLiteFTS5Retriever. Uses ts_rank_cd() for scoring
    with 'simple' text search config (no stemming — preserves
    maritime acronyms like SOLAS, MARPOL, BWM).
    """

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        similarity_top_k: int = 10,
    ):
        super().__init__()
        self.tenant_id = tenant_id
        self.similarity_top_k = similarity_top_k

        LOGGER.debug(
            "PgFTSRetriever initialized: tenant=%s, top_k=%d",
            tenant_id or "all", similarity_top_k
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str

        if not query_str.strip():
            return []

        tsquery = self._prepare_tsquery(query_str)
        LOGGER.info("PG FTS query: %s", tsquery)

        try:
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    if self.tenant_id:
                        cur.execute("""
                            SELECT
                                n.node_id,
                                n.text,
                                n.metadata,
                                n.doc_id,
                                n.section_id,
                                ts_rank_cd(n.tsv, to_tsquery('simple', %s)) AS score
                            FROM nodes n
                            WHERE n.tsv @@ to_tsquery('simple', %s)
                              AND (n.tenant_id = %s OR n.tenant_id = 'shared')
                            ORDER BY score DESC
                            LIMIT %s
                        """, (tsquery, tsquery, self.tenant_id, self.similarity_top_k))
                    else:
                        cur.execute("""
                            SELECT
                                n.node_id,
                                n.text,
                                n.metadata,
                                n.doc_id,
                                n.section_id,
                                ts_rank_cd(n.tsv, to_tsquery('simple', %s)) AS score
                            FROM nodes n
                            WHERE n.tsv @@ to_tsquery('simple', %s)
                            ORDER BY score DESC
                            LIMIT %s
                        """, (tsquery, tsquery, self.similarity_top_k))

                    rows = cur.fetchall()

            LOGGER.info("PG FTS returned %d results", len(rows))

            results = []
            for row in rows:
                node = self._row_to_node(row)
                # ts_rank_cd returns positive scores (higher = better)
                score = float(row["score"]) if row["score"] else 0.0
                results.append(NodeWithScore(node=node, score=score))

            return results

        except Exception as exc:
            LOGGER.error("PG FTS search failed: %s", exc)
            return []

    def retrieve_filtered(
        self,
        query_str: str,
        doc_type_filter: Optional[List[str]] = None,
        title_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        if not query_str.strip():
            return []

        effective_top_k = top_k or self.similarity_top_k
        retrieval_top_k = effective_top_k * 2 if title_filter else effective_top_k

        tsquery = self._prepare_tsquery(query_str)

        try:
            with pg_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query
                    conditions = ["n.tsv @@ to_tsquery('simple', %s)"]
                    params: list = [tsquery]

                    if self.tenant_id:
                        conditions.append("(n.tenant_id = %s OR n.tenant_id = 'shared')")
                        params.append(self.tenant_id)

                    if doc_type_filter:
                        expanded = _expand_doc_type_case(doc_type_filter)
                        conditions.append("n.metadata->>'doc_type' = ANY(%s)")
                        params.append(expanded)

                    where_clause = " AND ".join(conditions)
                    params.extend([tsquery, retrieval_top_k])

                    sql = f"""
                        SELECT
                            n.node_id, n.text, n.metadata,
                            n.doc_id, n.section_id,
                            ts_rank_cd(n.tsv, to_tsquery('simple', %s)) AS score
                        FROM nodes n
                        WHERE {where_clause}
                        ORDER BY score DESC
                        LIMIT %s
                    """

                    cur.execute(sql, params)
                    rows = cur.fetchall()

            LOGGER.info(
                "PG FTS filtered: %d results (doc_type=%s, title=%s)",
                len(rows), doc_type_filter, title_filter,
            )

            results = []
            for row in rows:
                node = self._row_to_node(row)
                score = float(row["score"]) if row["score"] else 0.0
                results.append(NodeWithScore(node=node, score=score))

            # Post-retrieval title filter (same as before)
            if title_filter:
                before = len(results)
                pattern = title_filter.upper()
                results = [
                    r for r in results
                    if pattern in (r.node.metadata.get("title") or "").upper()
                ]
                LOGGER.info(
                    "Title filter '%s': %d → %d results",
                    title_filter, before, len(results),
                )

            return results[:effective_top_k]

        except Exception as exc:
            LOGGER.error("PG FTS filtered search failed: %s", exc)
            return []

    def _prepare_tsquery(self, query: str) -> str:
        """Prepare query string for PostgreSQL to_tsquery().

        Splits into words, joins with | (OR) for broad matching.
        Uses 'simple' config — no stemming, preserves acronyms.

        Returns a string safe for to_tsquery('simple', ...).
        """
        words = query.strip().split()

        if not words:
            return ""

        # Clean each word: remove characters that break tsquery syntax
        cleaned = []
        for word in words:
            # Strip anything that isn't alphanumeric, hyphen, or period
            clean = "".join(c for c in word if c.isalnum() or c in "-.")
            if clean:
                cleaned.append(clean)

        if not cleaned:
            return ""

        # Join with | (OR) — matches documents containing ANY term
        return " | ".join(cleaned)

    def _row_to_node(self, row) -> TextNode:
        metadata = row["metadata"] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        metadata.setdefault("source", row["doc_id"])
        if row.get("section_id"):
            metadata.setdefault("section_id", row["section_id"])

        return TextNode(
            id_=row["node_id"],
            text=row["text"],
            metadata=metadata,
        )


# =============================================================================
# TENANT-AWARE VECTOR RETRIEVER (Phase 4 — Qdrant)
# =============================================================================

class TenantAwareVectorRetriever(BaseRetriever):
    """Vector retriever with tenant isolation via Qdrant payload filtering.

    Queries Qdrant directly (not through LlamaIndex) for full control
    over filter composition, scoring, and result construction.
    Returns vectors belonging to the specified tenant OR 'shared'.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embed_model,
        tenant_id: Optional[str] = None,
        similarity_top_k: int = 10,
    ):
        super().__init__()
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.tenant_id = tenant_id
        self.similarity_top_k = similarity_top_k

        LOGGER.debug(
            "TenantAwareVectorRetriever initialized: tenant=%s, top_k=%d, collection=%s",
            tenant_id or "all", similarity_top_k, collection_name,
        )

    # ------------------------------------------------------------------
    # Filter builders
    # ------------------------------------------------------------------

    def _build_tenant_filter(self) -> Optional[Filter]:
        """Build a Qdrant filter for tenant scoping (tenant OR shared)."""
        if not self.tenant_id:
            return None
        return Filter(should=[
            FieldCondition(key="tenant_id", match=MatchValue(value=self.tenant_id)),
            FieldCondition(key="tenant_id", match=MatchValue(value="shared")),
        ])

    @staticmethod
    def _build_doc_type_condition(doc_type_filter: List[str]) -> FieldCondition:
        """Build a FieldCondition for doc_type matching."""
        expanded = _expand_doc_type_case(doc_type_filter)
        if len(expanded) == 1:
            return FieldCondition(key="doc_type", match=MatchValue(value=expanded[0]))
        return FieldCondition(key="doc_type", match=MatchAny(any=expanded))

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _points_to_nodes(points) -> List[NodeWithScore]:
        """Convert Qdrant ScoredPoints to LlamaIndex NodeWithScore list."""
        nodes = []
        for point in points:
            payload = point.payload or {}
            text = _extract_text_from_payload(payload)
            metadata = _extract_metadata_from_payload(payload)

            node = TextNode(
                id_=str(point.id),
                text=text,
                metadata=metadata,
            )
            # Qdrant cosine → similarity score directly (higher = better)
            nodes.append(NodeWithScore(node=node, score=point.score))
        return nodes

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve vectors filtered by tenant."""
        query_str = query_bundle.query_str
        if not query_str.strip():
            return []

        try:
            query_embedding = self.embed_model.get_query_embedding(query_str)
            tenant_filter = self._build_tenant_filter()

            result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=tenant_filter,
                limit=self.similarity_top_k,
                with_payload=True,
            )

            points = result.points
            LOGGER.info(
                "Vector search returned %d results (tenant=%s)",
                len(points), self.tenant_id or "all",
            )
            return self._points_to_nodes(points)

        except Exception as exc:
            LOGGER.error("Vector search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Extended API (used by orchestrator)
    # ------------------------------------------------------------------

    def retrieve_filtered(
        self,
        query_str: str,
        doc_type_filter: Optional[List[str]] = None,
        title_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """Retrieve vectors with optional metadata filters.

        Used by the orchestrator's FilteredRetriever for source-specific
        retrieval.

        Args:
            query_str: Raw search query.
            doc_type_filter: e.g. ``["REGULATION", "VETTING"]``.
            title_filter: Substring to match in document title
                (applied post-retrieval on metadata).
            top_k: Override the instance's ``similarity_top_k``.

        Returns:
            Filtered list of ``NodeWithScore`` ranked by similarity.
        """
        if not query_str.strip():
            return []

        effective_top_k = top_k or self.similarity_top_k
        # Over-fetch when title filtering is active (post-retrieval filter)
        retrieval_top_k = effective_top_k * 2 if title_filter else effective_top_k

        try:
            query_embedding = self.embed_model.get_query_embedding(query_str)

            # ---- build composite Qdrant filter ----
            must_conditions: List = []

            # Tenant scoping (OR logic wrapped in a nested Filter)
            tenant_filter = self._build_tenant_filter()
            if tenant_filter:
                must_conditions.append(tenant_filter)

            # Doc-type filter
            if doc_type_filter:
                must_conditions.append(self._build_doc_type_condition(doc_type_filter))

            query_filter = Filter(must=must_conditions) if must_conditions else None

            # ---- query Qdrant ----
            result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=retrieval_top_k,
                with_payload=True,
            )

            points = result.points
            LOGGER.info(
                "Vector filtered: %d results (doc_type=%s, title=%s)",
                len(points), doc_type_filter, title_filter,
            )

            nodes = self._points_to_nodes(points)

            # ---- post-retrieval title filter ----
            if title_filter:
                title_lower = title_filter.lower()
                nodes = [
                    n for n in nodes
                    if title_lower in n.node.metadata.get("title", "").lower()
                ][:effective_top_k]

            return nodes

        except Exception as exc:
            LOGGER.error("Vector filtered search failed: %s", exc)
            return []


# =============================================================================
# POSTGRESQL NODE LOADER (Utility)
# =============================================================================

class PgNodeLoader:
    """Load nodes from PostgreSQL for operations that need full node sets.

    Use sparingly — prefer NodeRepository for scoped queries.
    """

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id

    def load_all_nodes(self) -> List[TextNode]:
        with pg_connection() as conn:
            with conn.cursor() as cur:
                if self.tenant_id:
                    cur.execute("""
                        SELECT node_id, text, metadata, doc_id, section_id
                        FROM nodes
                        WHERE tenant_id = %s OR tenant_id = 'shared'
                        ORDER BY doc_id, node_id
                    """, (self.tenant_id,))
                else:
                    cur.execute("""
                        SELECT node_id, text, metadata, doc_id, section_id
                        FROM nodes
                        ORDER BY doc_id, node_id
                    """)
                rows = cur.fetchall()

        LOGGER.info("Loaded %d nodes from PostgreSQL", len(rows))
        return [self._row_to_node(row) for row in rows]

    def _row_to_node(self, row) -> TextNode:
        """Convert database row to TextNode."""
        metadata = row["metadata"] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        metadata.setdefault("source", row["doc_id"])
        if row.get("section_id"):
            metadata.setdefault("section_id", row["section_id"])

        return TextNode(
            id_=row["node_id"],
            text=row["text"],
            metadata=metadata,
        )


__all__ = [
    "PgFTSRetriever",
    "TenantAwareVectorRetriever",
    "PgNodeLoader",
]