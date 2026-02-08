"""SQLite FTS5-based retriever and tenant-aware vector retriever.

Phase 3: FTS5 for keyword search (replaces in-memory BM25)
Phase 4: Tenant-aware vector retrieval (filters ChromaDB by tenant)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from .database import db_connection, get_db_path
from .logger import LOGGER


def _expand_doc_type_case(doc_type_filter: List[str]) -> List[str]:
    """Expand doc_type filter values to cover common case variants.

    Production data contains mixed casing (e.g. ``"Form"`` vs ``"FORM"``,
    ``"Procedure"`` vs ``"PROCEDURE"``).  Rather than requiring a full
    reindex, this helper emits both UPPER and Title variants for each
    requested type so that DB/ChromaDB filters match regardless of how
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
# FTS5 RETRIEVER (Phase 3)
# =============================================================================

class SQLiteFTS5Retriever(BaseRetriever):
    """
    Retriever that uses SQLite FTS5 for keyword search.
    
    Implements the same interface as BM25Retriever but queries
    the database instead of searching in-memory nodes.
    
    FTS5 uses BM25 ranking by default, so search quality is comparable.
    """
    
    def __init__(
        self,
        tenant_id: Optional[str] = None,
        similarity_top_k: int = 10,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize the FTS5 retriever.
        
        Args:
            tenant_id: If provided, restrict search to this tenant's nodes + shared.
                      If None, searches all nodes (for backward compatibility).
            similarity_top_k: Number of results to return.
            db_path: Path to SQLite database.
        """
        super().__init__()
        self.tenant_id = tenant_id
        self.similarity_top_k = similarity_top_k
        self.db_path = db_path or get_db_path()
        
        LOGGER.debug(
            "FTS5Retriever initialized: tenant=%s, top_k=%d",
            tenant_id or "all", similarity_top_k
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes matching the query using FTS5.
        
        Args:
            query_bundle: Query containing the search string.
        
        Returns:
            List of NodeWithScore objects ranked by BM25.
        """
        query_str = query_bundle.query_str
        
        if not query_str.strip():
            return []
        
        # Escape special FTS5 characters and prepare query
        fts_query = self._prepare_fts_query(query_str)
        LOGGER.info("FTS5 query: %s", fts_query)
        
        try:
            with db_connection(self.db_path) as conn:
                if self.tenant_id:
                    # Tenant-scoped search (includes shared docs)
                    cursor = conn.execute("""
                        SELECT 
                            n.node_id,
                            n.text,
                            n.metadata,
                            n.doc_id,
                            n.section_id,
                            bm25(nodes_fts) as score
                        FROM nodes_fts 
                        JOIN nodes n ON nodes_fts.rowid = n.rowid
                        WHERE nodes_fts MATCH ?
                          AND (n.tenant_id = ? OR n.tenant_id = 'shared')
                        ORDER BY score
                        LIMIT ?
                    """, (fts_query, self.tenant_id, self.similarity_top_k))
                else:
                    # Global search (all tenants)
                    cursor = conn.execute("""
                        SELECT 
                            n.node_id,
                            n.text,
                            n.metadata,
                            n.doc_id,
                            n.section_id,
                            bm25(nodes_fts) as score
                        FROM nodes_fts 
                        JOIN nodes n ON nodes_fts.rowid = n.rowid
                        WHERE nodes_fts MATCH ?
                        ORDER BY score
                        LIMIT ?
                    """, (fts_query, self.similarity_top_k))
                
                rows = cursor.fetchall()
            
            LOGGER.info("FTS5 returned %d results", len(rows))
            
            results = []
            for row in rows:
                node = self._row_to_node(row)
                # FTS5 bm25() returns negative scores (lower = better)
                # Normalize to positive (higher = better) for consistency
                score = -row["score"] if row["score"] else 0.0
                results.append(NodeWithScore(node=node, score=score))
            
            LOGGER.debug(
                "FTS5 search '%s': %d results (tenant=%s)",
                query_str[:50], len(results), self.tenant_id or "all"
            )
            
            return results
            
        except Exception as exc:
            LOGGER.error("FTS5 search failed: %s", exc)
            return []
    
    def retrieve_filtered(
        self,
        query_str: str,
        doc_type_filter: Optional[List[str]] = None,
        title_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """Retrieve with optional metadata filters.

        Used by the orchestrator's FilteredRetriever for source-specific
        retrieval.  Falls back to standard behaviour when no filters are
        provided.

        Args:
            query_str: Raw search query.
            doc_type_filter: e.g. ``["REGULATION", "VETTING"]``.
            title_filter: Substring to match in the document title
                          (applied post-retrieval on metadata).
            top_k: Override the instance's ``similarity_top_k``.

        Returns:
            Filtered list of ``NodeWithScore`` ranked by BM25.
        """
        if not query_str.strip():
            return []

        effective_top_k = top_k or self.similarity_top_k

        # When title filtering is active, retrieve more to compensate for
        # post-retrieval filtering losses.
        retrieval_top_k = effective_top_k * 2 if title_filter else effective_top_k

        fts_query = self._prepare_fts_query(query_str)

        try:
            with db_connection(self.db_path) as conn:
                # ---- build query dynamically ----
                conditions = ["nodes_fts MATCH ?"]
                params: list = [fts_query]

                if self.tenant_id:
                    conditions.append("(n.tenant_id = ? OR n.tenant_id = 'shared')")
                    params.append(self.tenant_id)

                if doc_type_filter:
                    expanded = _expand_doc_type_case(doc_type_filter)
                    placeholders = ", ".join("?" for _ in expanded)
                    conditions.append(
                        f"json_extract(n.metadata, '$.doc_type') IN ({placeholders})"
                    )
                    params.extend(expanded)

                where_clause = " AND ".join(conditions)
                params.append(retrieval_top_k)

                sql = f"""
                    SELECT
                        n.node_id, n.text, n.metadata,
                        n.doc_id, n.section_id,
                        bm25(nodes_fts) AS score
                    FROM nodes_fts
                    JOIN nodes n ON nodes_fts.rowid = n.rowid
                    WHERE {where_clause}
                    ORDER BY score
                    LIMIT ?
                """

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

            LOGGER.info(
                "FTS5 filtered: %d results (doc_type=%s, title=%s)",
                len(rows),
                doc_type_filter,
                title_filter,
            )

            results = []
            for row in rows:
                node = self._row_to_node(row)
                score = -row["score"] if row["score"] else 0.0
                results.append(NodeWithScore(node=node, score=score))

            # Post-retrieval title filter
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
            LOGGER.error("FTS5 filtered search failed: %s", exc)
            return []

    def _prepare_fts_query(self, query: str) -> str:
        """
        Prepare query string for FTS5 MATCH.
        
        FTS5 has special syntax characters that need escaping.
        We use a simple approach: split into words, quote each.
        
        Args:
            query: Raw query string
        
        Returns:
            FTS5-safe query string
        """
        # Remove special FTS5 operators that could cause syntax errors
        # Keep it simple: split words, join with OR for broader matching
        words = query.strip().split()
        
        if not words:
            return '""'  # Empty query
        
        # Escape each word by wrapping in quotes, join with OR
        # This matches documents containing ANY of the words
        escaped_words = []
        for word in words:
            # Remove any quotes from the word itself
            clean_word = word.replace('"', '').replace("'", "")
            if clean_word:
                escaped_words.append(f'"{clean_word}"')
        
        if not escaped_words:
            return '""'
        
        # Use OR for broader matching (same behavior as typical BM25)
        return " OR ".join(escaped_words)
    
    def _row_to_node(self, row: Any) -> TextNode:
        """
        Convert a database row to a TextNode.
        
        Args:
            row: SQLite Row object
        
        Returns:
            TextNode with metadata restored
        """
        # Parse metadata JSON
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse metadata for node %s", row["node_id"])
        
        # Ensure core fields are in metadata
        metadata.setdefault("source", row["doc_id"])
        if row["section_id"]:
            metadata.setdefault("section_id", row["section_id"])
        
        return TextNode(
            id_=row["node_id"],
            text=row["text"],
            metadata=metadata,
        )


# =============================================================================
# TENANT-AWARE VECTOR RETRIEVER (Phase 4)
# =============================================================================

class TenantAwareVectorRetriever(BaseRetriever):
    """
    Vector retriever with tenant isolation.
    
    Wraps ChromaDB queries to filter by tenant_id metadata.
    Returns vectors belonging to the specified tenant OR 'shared'.
    """
    
    def __init__(
        self,
        collection,
        embed_model,
        tenant_id: Optional[str] = None,
        similarity_top_k: int = 10,
    ):
        """
        Initialize tenant-aware vector retriever.
        
        Args:
            collection: ChromaDB collection
            embed_model: Embedding model for query encoding
            tenant_id: Tenant to filter for (also includes 'shared')
            similarity_top_k: Number of results to return
        """
        super().__init__()
        self.collection = collection
        self.embed_model = embed_model
        self.tenant_id = tenant_id
        self.similarity_top_k = similarity_top_k
        
        LOGGER.debug(
            "TenantAwareVectorRetriever initialized: tenant=%s, top_k=%d",
            tenant_id or "all", similarity_top_k
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve vectors filtered by tenant.
        
        Args:
            query_bundle: Query containing the search string
        
        Returns:
            List of NodeWithScore objects ranked by similarity
        """
        query_str = query_bundle.query_str
        
        if not query_str.strip():
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embed_model.get_query_embedding(query_str)
            
            # Build tenant filter
            where_filter = None
            if self.tenant_id:
                where_filter = {
                    "$or": [
                        {"tenant_id": self.tenant_id},
                        {"tenant_id": "shared"}
                    ]
                }
            
            # Query ChromaDB
            if where_filter:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.similarity_top_k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.similarity_top_k,
                    include=["documents", "metadatas", "distances"]
                )
            
            LOGGER.info(
                "Vector search returned %d results (tenant=%s)",
                len(results["ids"][0]) if results["ids"] else 0,
                self.tenant_id or "all"
            )
            
            # Convert to NodeWithScore
            nodes = []
            if results["ids"] and results["ids"][0]:
                for i, node_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    text = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    
                    node = TextNode(
                        id_=node_id,
                        text=text,
                        metadata=metadata,
                    )
                    
                    # Convert distance to similarity score (lower distance = higher similarity)
                    # ChromaDB returns L2 distance by default
                    score = 1.0 / (1.0 + distance)
                    
                    nodes.append(NodeWithScore(node=node, score=score))
            
            return nodes
            
        except Exception as exc:
            LOGGER.error("Vector search failed: %s", exc)
            return []


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
                Added to the ChromaDB ``where`` clause via ``$in``.
            title_filter: Substring to match in document title
                (applied post-retrieval on metadata).
            top_k: Override the instance's ``similarity_top_k``.

        Returns:
            Filtered list of ``NodeWithScore`` ranked by similarity.
        """
        if not query_str.strip():
            return []

        effective_top_k = top_k or self.similarity_top_k
        retrieval_top_k = effective_top_k * 2 if title_filter else effective_top_k

        try:
            query_embedding = self.embed_model.get_query_embedding(query_str)

            # ---- build where filter ----
            filter_parts: List[Dict] = []

            if self.tenant_id:
                filter_parts.append(
                    {"$or": [
                        {"tenant_id": self.tenant_id},
                        {"tenant_id": "shared"},
                    ]}
                )

            if doc_type_filter:
                expanded = _expand_doc_type_case(doc_type_filter)
                if len(expanded) == 1:
                    filter_parts.append({"doc_type": expanded[0]})
                else:
                    filter_parts.append({"doc_type": {"$in": expanded}})

            if len(filter_parts) == 0:
                where_filter = None
            elif len(filter_parts) == 1:
                where_filter = filter_parts[0]
            else:
                where_filter = {"$and": filter_parts}

            # ---- query ChromaDB ----
            query_kwargs = dict(
                query_embeddings=[query_embedding],
                n_results=retrieval_top_k,
                include=["documents", "metadatas", "distances"],
            )
            if where_filter:
                query_kwargs["where"] = where_filter

            results = self.collection.query(**query_kwargs)

            LOGGER.info(
                "Vector filtered: %d results (doc_type=%s, title=%s)",
                len(results["ids"][0]) if results["ids"] else 0,
                doc_type_filter,
                title_filter,
            )

            # ---- convert to NodeWithScore ----
            nodes = []
            if results["ids"] and results["ids"][0]:
                for i, node_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    text = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    node = TextNode(id_=node_id, text=text, metadata=metadata)
                    score = 1.0 / (1.0 + distance)
                    nodes.append(NodeWithScore(node=node, score=score))

            # Post-retrieval title filter
            if title_filter:
                before = len(nodes)
                pattern = title_filter.upper()
                nodes = [
                    n for n in nodes
                    if pattern in (n.node.metadata.get("title") or "").upper()
                ]
                LOGGER.info(
                    "Title filter '%s': %d → %d results",
                    title_filter, before, len(nodes),
                )

            return nodes[:effective_top_k]

        except Exception as exc:
            LOGGER.error("Vector filtered search failed: %s", exc)
            return []


# =============================================================================
# SQLITE NODE LOADER (Utility)
# =============================================================================

class SQLiteNodeLoader:
    """
    Helper class for loading nodes from SQLite.
    
    Used for operations that need full node objects (like hierarchical retrieval)
    but don't need the full in-memory cache.
    """
    
    def __init__(self, tenant_id: Optional[str] = None, db_path: Optional[Path] = None):
        self.tenant_id = tenant_id
        self.db_path = db_path or get_db_path()
    
    def get_node_by_id(self, node_id: str) -> Optional[TextNode]:
        """Load a single node by ID."""
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, text, metadata, doc_id, section_id
                FROM nodes
                WHERE node_id = ?
            """, (node_id,))
            row = cursor.fetchone()
        
        if row:
            return self._row_to_node(row)
        return None
    
    def get_nodes_by_doc(self, doc_id: str) -> List[TextNode]:
        """Load all nodes for a document."""
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, text, metadata, doc_id, section_id
                FROM nodes
                WHERE doc_id = ?
                ORDER BY node_id
            """, (doc_id,))
            rows = cursor.fetchall()
        
        return [self._row_to_node(row) for row in rows]
    
    def get_nodes_by_section(self, doc_id: str, section_id: str) -> List[TextNode]:
        """Load nodes for a specific section (for hierarchical retrieval)."""
        with db_connection(self.db_path) as conn:
            # Match section_id prefix for hierarchical sections
            # e.g., section_id "3.1" matches "3.1", "3.1.1", "3.1.2", etc.
            cursor = conn.execute("""
                SELECT node_id, text, metadata, doc_id, section_id
                FROM nodes
                WHERE doc_id = ?
                  AND (section_id = ? OR section_id LIKE ?)
                ORDER BY section_id, node_id
            """, (doc_id, section_id, f"{section_id}.%"))
            rows = cursor.fetchall()
        
        return [self._row_to_node(row) for row in rows]
    
    def get_all_nodes(self) -> List[TextNode]:
        """
        Load all nodes for the tenant.
        
        Use sparingly - for operations that truly need all nodes.
        """
        with db_connection(self.db_path) as conn:
            if self.tenant_id:
                cursor = conn.execute("""
                    SELECT node_id, text, metadata, doc_id, section_id
                    FROM nodes
                    WHERE tenant_id = ? OR tenant_id = 'shared'
                    ORDER BY doc_id, node_id
                """, (self.tenant_id,))
            else:
                cursor = conn.execute("""
                    SELECT node_id, text, metadata, doc_id, section_id
                    FROM nodes
                    ORDER BY doc_id, node_id
                """)
            rows = cursor.fetchall()
        
        LOGGER.info("Loaded %d nodes from SQLite", len(rows))
        return [self._row_to_node(row) for row in rows]
    
    def _row_to_node(self, row: Any) -> TextNode:
        """Convert database row to TextNode."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
        
        metadata.setdefault("source", row["doc_id"])
        if row["section_id"]:
            metadata.setdefault("section_id", row["section_id"])
        
        return TextNode(
            id_=row["node_id"],
            text=row["text"],
            metadata=metadata,
        )


__all__ = [
    "SQLiteFTS5Retriever", 
    "TenantAwareVectorRetriever",
    "SQLiteNodeLoader"
]